#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace triton;

namespace {

// Constant for Cache Tile Size
constexpr int64_t CACHE_TILE_K = 256;

// Helper to convert memref to triton pointer
Value memrefToPtr(PatternRewriter &rewriter, Location loc, Value memref) {
    auto memRefType = mlir::cast<MemRefType>(memref.getType());
    Type elemType = memRefType.getElementType();
    
    Value basePtrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, memref);
    Value basePtrInt = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), basePtrIndex);
    
    Type ptrType = triton::PointerType::get(elemType, 0);
    return rewriter.create<triton::IntToPtrOp>(loc, ptrType, basePtrInt);
}

// Optimization: Generate pointers for a contiguous row-major block [Rows, Cols]
// Offset = Base + (RowIdx * Stride) + ColIdx
Value createBlockPointers(PatternRewriter &rewriter, Location loc, Value basePtr, ArrayRef<int64_t> shape, int64_t stride) {
    int64_t rows = shape[0];
    int64_t cols = shape[1];
    
    // 1. Create Range(0, Rows) and Range(0, Cols)
    Value rangeRows = rewriter.create<triton::MakeRangeOp>(loc, RankedTensorType::get({rows}, rewriter.getI32Type()), 0, rows);
    Value rangeCols = rewriter.create<triton::MakeRangeOp>(loc, RankedTensorType::get({cols}, rewriter.getI32Type()), 0, cols);
    
    // 2. Expand and Broadcast to [Rows, Cols]
    Value rows2D = rewriter.create<triton::ExpandDimsOp>(loc, rangeRows, 1);
    rows2D = rewriter.create<triton::BroadcastOp>(loc, RankedTensorType::get({rows, cols}, rewriter.getI32Type()), rows2D);
    
    Value cols2D = rewriter.create<triton::ExpandDimsOp>(loc, rangeCols, 0);
    cols2D = rewriter.create<triton::BroadcastOp>(loc, RankedTensorType::get({rows, cols}, rewriter.getI32Type()), cols2D);
    
    // 3. Calculate Linear Offsets: (Row * Stride) + Col
    Value cStride = rewriter.create<arith::ConstantIntOp>(loc, stride, 32);
    Value strideSplat = rewriter.create<triton::SplatOp>(loc, RankedTensorType::get({rows, cols}, rewriter.getI32Type()), cStride);
    
    Value rowOffset = rewriter.create<arith::MulIOp>(loc, rows2D, strideSplat);
    Value totalOffset = rewriter.create<arith::AddIOp>(loc, rowOffset, cols2D);
    
    // 4. Add to Base Pointer
    Value baseSplat = rewriter.create<triton::SplatOp>(loc, RankedTensorType::get({rows, cols}, basePtr.getType()), basePtr);
    
    return rewriter.create<triton::AddPtrOp>(loc, baseSplat.getType(), baseSplat, totalOffset);
}

// Helper to store a tensor to a memref (pointer)
void storeTensorToMemRef(PatternRewriter &rewriter, Location loc, Value tensor, Value memref, Value offsetK, bool isA) {
    auto tensorType = mlir::cast<RankedTensorType>(tensor.getType());
    auto shape = tensorType.getShape();
    
    // Get packed width from memref shape
    auto memRefType = mlir::cast<MemRefType>(memref.getType());
    int64_t packedWidth = memRefType.getShape()[1];
    
    Value basePtr = memrefToPtr(rewriter, loc, memref);
    
    // Calculate Base Offset
    Value ptrOffset;
    if (isA) {
        // For A (MxK), we are storing a block at offsetK in the K dimension (dim 1).
        // Base Offset = offsetK
        // offsetK is likely i32 or index, cast to i64 for pointer arithmetic
        if (offsetK.getType().isIndex()) {
             ptrOffset = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), offsetK);
        } else {
             ptrOffset = rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), offsetK);
        }
    } else {
        // For B (KxN), we are storing a block at offsetK in the K dimension (dim 0).
        // Base Offset = offsetK * packedWidth (Stride)
        Value offsetK_i64;
        if (offsetK.getType().isIndex()) {
             offsetK_i64 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), offsetK);
        } else {
             offsetK_i64 = rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), offsetK);
        }
        Value width_i64 = rewriter.create<arith::ConstantIntOp>(loc, packedWidth, 64);
        ptrOffset = rewriter.create<arith::MulIOp>(loc, offsetK_i64, width_i64);
    }
    
    // Adjust Base Pointer
    Type ptrType = basePtr.getType();
    Value offsetPtr = rewriter.create<triton::AddPtrOp>(loc, ptrType, basePtr, ptrOffset);

    // Generate Pointers
    Value ptrs = createBlockPointers(rewriter, loc, offsetPtr, shape, packedWidth);
    
    rewriter.create<triton::StoreOp>(loc, ptrs, tensor, triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);
}

// Helper to load a tensor from a memref (pointer)
Value loadTensorFromMemRef(PatternRewriter &rewriter, Location loc, Value memref, Value offsetK, RankedTensorType tensorType, bool isA) {
    auto shape = tensorType.getShape();
    
    // Get packed width from memref shape
    auto memRefType = mlir::cast<MemRefType>(memref.getType());
    int64_t packedWidth = memRefType.getShape()[1];
    
    Value basePtr = memrefToPtr(rewriter, loc, memref);
    
    // Calculate Base Offset
    Value ptrOffset;
    if (isA) {
        // offsetK is likely i32 or index, cast to i64 for pointer arithmetic
        if (offsetK.getType().isIndex()) {
             ptrOffset = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), offsetK);
        } else {
             ptrOffset = rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), offsetK);
        }
    } else {
        Value offsetK_i64;
        if (offsetK.getType().isIndex()) {
             offsetK_i64 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), offsetK);
        } else {
             offsetK_i64 = rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), offsetK);
        }
        Value width_i64 = rewriter.create<arith::ConstantIntOp>(loc, packedWidth, 64);
        ptrOffset = rewriter.create<arith::MulIOp>(loc, offsetK_i64, width_i64);
    }
    
    // Adjust Base Pointer
    Type ptrType = basePtr.getType();
    Value offsetPtr = rewriter.create<triton::AddPtrOp>(loc, ptrType, basePtr, ptrOffset);

    // Generate Pointers
    Value ptrs = createBlockPointers(rewriter, loc, offsetPtr, shape, packedWidth);
    
    return rewriter.create<triton::LoadOp>(loc, ptrs, triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL, false);
}

struct BlockPackingPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp, PatternRewriter &rewriter) const override {
    auto dotOps = forOp.getOps<triton::DotOp>();
    if (dotOps.empty()) return failure();
    triton::DotOp dotOp = *dotOps.begin();

    Value lb = forOp.getLowerBound();
    Value ub = forOp.getUpperBound();
    Value step = forOp.getStep();

    if (!step) return failure();
    
    int64_t blockK = 0;
    Operation *defOp = step.getDefiningOp();
    if (auto constOp = dyn_cast_or_null<arith::ConstantIndexOp>(defOp)) {
        blockK = constOp.value();
    } else if (auto constOp = dyn_cast_or_null<arith::ConstantIntOp>(defOp)) {
        blockK = constOp.value();
    } else if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(defOp)) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
            blockK = intAttr.getInt();
        } else {
            return failure();
        }
    } else {
        return failure();
    }
    


    if (blockK >= CACHE_TILE_K) return failure();
    if (blockK <= 0) return failure();

    // Check if loop range is small enough to skip tiling
    // This prevents infinite recursion when the pass runs on the generated inner loop
    int64_t lbVal = -1, ubVal = -1;
    if (auto constLb = dyn_cast_or_null<arith::ConstantIndexOp>(lb.getDefiningOp())) lbVal = constLb.value();
    else if (auto constLb = dyn_cast_or_null<arith::ConstantIntOp>(lb.getDefiningOp())) lbVal = constLb.value();
    
    if (auto constUb = dyn_cast_or_null<arith::ConstantIndexOp>(ub.getDefiningOp())) ubVal = constUb.value();
    else if (auto constUb = dyn_cast_or_null<arith::ConstantIntOp>(ub.getDefiningOp())) ubVal = constUb.value();
    
    if (lbVal != -1 && ubVal != -1) {
        int64_t diff = ubVal - lbVal;
        if (diff < 0) diff = -diff;
        if (diff <= CACHE_TILE_K) return failure();
    }

    // Identify A and B
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto loadA = a.getDefiningOp<triton::LoadOp>();
    auto loadB = b.getDefiningOp<triton::LoadOp>();
    
    if (!loadA || !loadB) return failure();

    // Create Outer Loop
    Type loopType = step.getType();
    Value cCacheTileK;
    if (loopType.isIndex()) {
        cCacheTileK = rewriter.create<arith::ConstantIndexOp>(forOp.getLoc(), CACHE_TILE_K);
    } else {
        cCacheTileK = rewriter.create<arith::ConstantIntOp>(forOp.getLoc(), CACHE_TILE_K, loopType);
    }

    SmallVector<Value> iterArgs = forOp.getInitArgs();
    auto outerLoop = rewriter.create<scf::ForOp>(
        forOp.getLoc(), lb, ub, 
        cCacheTileK,
        iterArgs);
    
    rewriter.setInsertionPointToStart(outerLoop.getBody());
    Value kOuter = outerLoop.getInductionVar();

    auto tensorTypeA = mlir::cast<RankedTensorType>(a.getType());
    auto tensorTypeB = mlir::cast<RankedTensorType>(b.getType());
    
    // Allocate Scratchpads
    SmallVector<int64_t> packedShapeA = {tensorTypeA.getShape()[0], CACHE_TILE_K};
    auto memRefTypeA = MemRefType::get(packedShapeA, tensorTypeA.getElementType());
    Value packedA = rewriter.create<memref::AllocaOp>(forOp.getLoc(), memRefTypeA);
    
    SmallVector<int64_t> packedShapeB = {CACHE_TILE_K, tensorTypeB.getShape()[1]};
    auto memRefTypeB = MemRefType::get(packedShapeB, tensorTypeB.getElementType());
    Value packedB = rewriter.create<memref::AllocaOp>(forOp.getLoc(), memRefTypeB);

    // Packing Loop
    Value c0;
    if (loopType.isIndex()) {
        c0 = rewriter.create<arith::ConstantIndexOp>(forOp.getLoc(), 0);
    } else {
        c0 = rewriter.create<arith::ConstantIntOp>(forOp.getLoc(), 0, loopType);
    }
    // cCacheTileK is already created with correct type
    Value cBlockK;
    if (loopType.isIndex()) {
        cBlockK = rewriter.create<arith::ConstantIndexOp>(forOp.getLoc(), blockK);
    } else {
        cBlockK = rewriter.create<arith::ConstantIntOp>(forOp.getLoc(), blockK, loopType);
    }
    
    auto packingLoop = rewriter.create<scf::ForOp>(
        forOp.getLoc(), c0, cCacheTileK, cBlockK, outerLoop.getRegionIterArgs());
        
    rewriter.setInsertionPointToStart(packingLoop.getBody());
    Value kInnerPack = packingLoop.getInductionVar();
    Value globalK = rewriter.create<arith::AddIOp>(forOp.getLoc(), kOuter, kInnerPack);
    
    // Find Accumulator Index
    auto regionIterArgs = forOp.getRegionIterArgs();
    int accIdx = -1;
    for (int i = 0; i < regionIterArgs.size(); ++i) {
        if (regionIterArgs[i] == dotOp.getOperand(2)) {
            accIdx = i;
            break;
        }
    }
    if (accIdx == -1) return failure();

    // Clone Loads
    IRMapping mapping;
    mapping.map(forOp.getInductionVar(), globalK);
    for (int i = 0; i < regionIterArgs.size(); ++i) {
        mapping.map(regionIterArgs[i], packingLoop.getRegionIterArgs()[i]);
    }
    
    std::function<Value(Value)> cloneRecursive = [&](Value val) -> Value {
        if (mapping.contains(val)) return mapping.lookup(val);
        Operation* defOp = val.getDefiningOp();
        if (!defOp || defOp->getBlock() != forOp.getBody()) return val;
        
        IRMapping localMap;
        for (Value op : defOp->getOperands()) {
            localMap.map(op, cloneRecursive(op));
        }
        Operation* clonedOp = rewriter.clone(*defOp, localMap);
        for (unsigned i = 0; i < defOp->getNumResults(); ++i) {
            mapping.map(defOp->getResult(i), clonedOp->getResult(i));
        }
        for (unsigned i = 0; i < defOp->getNumResults(); ++i) {
            if (defOp->getResult(i) == val) return clonedOp->getResult(i);
        }
        return Value();
    };

    // Clone dependencies for LoadA
    for (Value op : loadA->getOperands()) {
        cloneRecursive(op);
    }
    Operation* newLoadA = rewriter.clone(*loadA, mapping);
    
    // Clone dependencies for LoadB
    for (Value op : loadB->getOperands()) {
        cloneRecursive(op);
    }
    Operation* newLoadB = rewriter.clone(*loadB, mapping);
    
    // Store to Packed
    storeTensorToMemRef(rewriter, forOp.getLoc(), newLoadA->getResult(0), packedA, kInnerPack, true);
    storeTensorToMemRef(rewriter, forOp.getLoc(), newLoadB->getResult(0), packedB, kInnerPack, false);
    
    // Yield updated args
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    SmallVector<Value> packingYieldValues;
    for (int i = 0; i < iterArgs.size(); ++i) {
        if (i == accIdx) {
            packingYieldValues.push_back(packingLoop.getRegionIterArgs()[i]);
        } else {
            Value nextVal = cloneRecursive(yieldOp.getOperand(i));
            packingYieldValues.push_back(nextVal);
        }
    }
    
    {
        Block *body = packingLoop.getBody();
        // llvm::errs() << "Packing Loop Body:\n";
        // for (auto &op : *body) llvm::errs() << "  " << op.getName() << "\n";
        
        if (!body->empty() && body->back().hasTrait<OpTrait::IsTerminator>()) {
            rewriter.replaceOpWithNewOp<scf::YieldOp>(&body->back(), packingYieldValues);
        } else {
            rewriter.create<scf::YieldOp>(forOp.getLoc(), packingYieldValues);
        }
    }
    
    rewriter.setInsertionPointAfter(packingLoop);

    // Inner Compute Loop
    auto innerLoop = rewriter.create<scf::ForOp>(
        forOp.getLoc(), c0, cCacheTileK, cBlockK, packingLoop.getResults());
    
    rewriter.setInsertionPointToStart(innerLoop.getBody());
    Value kInner = innerLoop.getInductionVar();
    Value accInner = innerLoop.getRegionIterArgs()[accIdx];
    
    // Load from Packed
    Value aFast = loadTensorFromMemRef(rewriter, forOp.getLoc(), packedA, kInner, tensorTypeA, true);
    Value bFast = loadTensorFromMemRef(rewriter, forOp.getLoc(), packedB, kInner, tensorTypeB, false);
    
    // Dot
    SmallVector<Value> newDotOperands;
    newDotOperands.push_back(aFast);
    newDotOperands.push_back(bFast);
    newDotOperands.push_back(accInner); // C
    
    auto newDot = rewriter.create<triton::DotOp>(forOp.getLoc(), dotOp.getType(), newDotOperands, dotOp->getAttrs());
    
    SmallVector<Value> innerYieldValues;
    for (int i = 0; i < iterArgs.size(); ++i) {
        if (i == accIdx) {
            innerYieldValues.push_back(newDot.getResult());
        } else {
            innerYieldValues.push_back(innerLoop.getRegionIterArgs()[i]);
        }
    }
    
    {
        Block *body = innerLoop.getBody();
        // llvm::errs() << "Inner Loop Body:\n";
        // for (auto &op : *body) llvm::errs() << "  " << op.getName() << "\n";

        if (!body->empty() && body->back().hasTrait<OpTrait::IsTerminator>()) {
            rewriter.replaceOpWithNewOp<scf::YieldOp>(&body->back(), innerYieldValues);
        } else {
            rewriter.create<scf::YieldOp>(forOp.getLoc(), innerYieldValues);
        }
    }
    
    rewriter.setInsertionPointAfter(innerLoop);
    
    {
        Block *body = outerLoop.getBody();
        // llvm::errs() << "Outer Loop Body:\n";
        // for (auto &op : *body) llvm::errs() << "  " << op.getName() << "\n";

        if (!body->empty() && body->back().hasTrait<OpTrait::IsTerminator>()) {
            rewriter.replaceOpWithNewOp<scf::YieldOp>(&body->back(), innerLoop.getResults());
        } else {
            rewriter.create<scf::YieldOp>(forOp.getLoc(), innerLoop.getResults());
        }
    }
    
    rewriter.replaceOp(forOp, outerLoop.getResults());
    
    return success();
  }
};
/*
    auto tensorTypeA = mlir::cast<RankedTensorType>(a.getType());
    auto tensorTypeB = mlir::cast<RankedTensorType>(b.getType());
    
    // Allocate Scratchpads
    SmallVector<int64_t> packedShapeA = {tensorTypeA.getShape()[0], CACHE_TILE_K};
    auto memRefTypeA = MemRefType::get(packedShapeA, tensorTypeA.getElementType());
    Value packedA = rewriter.create<memref::AllocaOp>(forOp.getLoc(), memRefTypeA);
    
    SmallVector<int64_t> packedShapeB = {CACHE_TILE_K, tensorTypeB.getShape()[1]};
    auto memRefTypeB = MemRefType::get(packedShapeB, tensorTypeB.getElementType());
    Value packedB = rewriter.create<memref::AllocaOp>(forOp.getLoc(), memRefTypeB);

    // Packing Loop
    Value c0;
    if (loopType.isIndex()) {
        c0 = rewriter.create<arith::ConstantIndexOp>(forOp.getLoc(), 0);
    } else {
        c0 = rewriter.create<arith::ConstantIntOp>(forOp.getLoc(), 0, loopType);
    }
    // cCacheTileK is already created with correct type
    Value cBlockK;
    if (loopType.isIndex()) {
        cBlockK = rewriter.create<arith::ConstantIndexOp>(forOp.getLoc(), blockK);
    } else {
        cBlockK = rewriter.create<arith::ConstantIntOp>(forOp.getLoc(), blockK, loopType);
    }
    
    auto packingLoop = rewriter.create<scf::ForOp>(
        forOp.getLoc(), c0, cCacheTileK, cBlockK);
        
    rewriter.setInsertionPointToStart(packingLoop.getBody());
    Value kInnerPack = packingLoop.getInductionVar();
    Value globalK = rewriter.create<arith::AddIOp>(forOp.getLoc(), kOuter, kInnerPack);
    
    // Clone Loads
    IRMapping mapping;
    mapping.map(forOp.getInductionVar(), globalK);
    
    std::function<Value(Value)> cloneRecursive = [&](Value val) -> Value {
        if (mapping.contains(val)) return mapping.lookup(val);
        Operation* defOp = val.getDefiningOp();
        if (!defOp || defOp->getBlock() != forOp.getBody()) return val;
        
        IRMapping localMap;
        for (Value op : defOp->getOperands()) {
            localMap.map(op, cloneRecursive(op));
        }
        Operation* clonedOp = rewriter.clone(*defOp, localMap);
        for (unsigned i = 0; i < defOp->getNumResults(); ++i) {
            mapping.map(defOp->getResult(i), clonedOp->getResult(i));
        }
        for (unsigned i = 0; i < defOp->getNumResults(); ++i) {
            if (defOp->getResult(i) == val) return clonedOp->getResult(i);
        }
        return Value();
    };

    // Clone dependencies for LoadA
    for (Value op : loadA->getOperands()) {
        cloneRecursive(op);
    }
    Operation* newLoadA = rewriter.clone(*loadA, mapping);
    
    // Clone dependencies for LoadB
    for (Value op : loadB->getOperands()) {
        cloneRecursive(op);
    }
    Operation* newLoadB = rewriter.clone(*loadB, mapping);
    
    // Store to Packed
    storeTensorToMemRef(rewriter, forOp.getLoc(), newLoadA->getResult(0), packedA, kInnerPack, true);
    storeTensorToMemRef(rewriter, forOp.getLoc(), newLoadB->getResult(0), packedB, kInnerPack, false);
    
    rewriter.create<scf::YieldOp>(forOp.getLoc());
    
    rewriter.setInsertionPointAfter(packingLoop);

    // Inner Compute Loop
    auto innerLoop = rewriter.create<scf::ForOp>(
        forOp.getLoc(), c0, cCacheTileK, cBlockK, outerLoop.getRegionIterArgs());
    
    rewriter.setInsertionPointToStart(innerLoop.getBody());
    Value kInner = innerLoop.getInductionVar();
    Value accInner = innerLoop.getRegionIterArgs()[0];
    
    // Load from Packed
    Value aFast = loadTensorFromMemRef(rewriter, forOp.getLoc(), packedA, kInner, tensorTypeA, true);
    Value bFast = loadTensorFromMemRef(rewriter, forOp.getLoc(), packedB, kInner, tensorTypeB, false);
    
    // Dot
    SmallVector<Value> newDotOperands;
    newDotOperands.push_back(aFast);
    newDotOperands.push_back(bFast);
    newDotOperands.push_back(accInner); // C
    
    auto newDot = rewriter.create<triton::DotOp>(forOp.getLoc(), dotOp.getType(), newDotOperands, dotOp->getAttrs());
    
    rewriter.create<scf::YieldOp>(forOp.getLoc(), newDot.getResult());
    
    rewriter.setInsertionPointAfter(innerLoop);
    rewriter.create<scf::YieldOp>(forOp.getLoc(), innerLoop.getResults());
    
    rewriter.replaceOp(forOp, outerLoop.getResults());
    
    return success();
  }
};
*/
class TritonCPUBlockPackingPass : public PassWrapper<TritonCPUBlockPackingPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TritonCPUBlockPackingPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<BlockPackingPattern>(context);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace triton {

std::unique_ptr<Pass> createTritonCPUBlockPackingPass() {
  return std::make_unique<TritonCPUBlockPackingPass>();
}

} // namespace triton
} // namespace mlir