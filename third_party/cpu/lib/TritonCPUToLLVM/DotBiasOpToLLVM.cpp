#include <memory> 

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

// Correctly use the Passes.h from TritonCPUToLLVM
#include "cpu/include/TritonCPUToLLVM/Passes.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_DOTBIASOPTOLLVM
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace{

// Define the pattern to convert 'DotBiasOp' -> Loops + Vector Ops
struct DotBiasToLoopsPattern : public ConvertOpToLLVMPattern<DotBiasOp> {
  
  // Standard constructor
  using ConvertOpToLLVMPattern<DotBiasOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(DotBiasOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    
    // 1. Get Accessors to Operands (Converted to MemRefs/Pointers by previous passes)
    Value aPtr = adaptor.getA();    // Pointer to Matrix A
    Value bPtr = adaptor.getB();    // Pointer to Matrix B
    Value biasPtr = adaptor.getBias(); // Pointer to Bias Vector
    Value resPtr = adaptor.getResult(); // Pointer to Output Matrix

    // 2. Get Shapes (Dimensions M, N, K)
    // We assume fixed shapes for simplicity here. In reality, you extract these from the type.
    auto aShape = op.getA().getType().cast<RankedTensorType>().getShape();
    int M = aShape[0];
    int K = aShape[1];
    int N = op.getB().getType().cast<RankedTensorType>().getShape()[1];

    Location loc = op.getLoc();
    
    // Constants for loops
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value ubM = rewriter.create<arith::ConstantIndexOp>(loc, M);
    Value ubN = rewriter.create<arith::ConstantIndexOp>(loc, N);
    Value ubK = rewriter.create<arith::ConstantIndexOp>(loc, K);

    // 3. Generate the Loop Nest (M, N, K)
    // Outer Loop: Rows (M)
    rewriter.create<scf::ForOp>(loc, c0, ubM, c1, [&](OpBuilder &b, Location loc, Value mIdx) {
        
        // Middle Loop: Cols (N)
        b.create<scf::ForOp>(loc, c0, ubN, c1, [&](OpBuilder &b, Location loc, Value nIdx) {
            
            // Initialize Accumulator with 0.0
            Value acc = b.create<arith::ConstantFloatOp>(loc, llvm::APFloat(0.0f), b.getF32Type());

            // Inner Loop: Reduction (K) - The Dot Product
            // We use 'iter_args' to carry the accumulator value 'currAcc'
            auto kLoop = b.create<scf::ForOp>(loc, c0, ubK, c1, ValueRange{acc}, 
                [&](OpBuilder &b, Location loc, Value kIdx, ValueRange args) {
                    
                    Value currAcc = args[0];
                    
                    // Load A[m, k]
                    // (Omitted: Index calculation math for simplicity)
                    Value valA = b.create<memref::LoadOp>(loc, aPtr, ValueRange{mIdx, kIdx});

                    // Load B[k, n]
                    Value valB = b.create<memref::LoadOp>(loc, bPtr, ValueRange{kIdx, nIdx});

                    // FMA: acc += valA * valB
                    Value mul = b.create<arith::MulFOp>(loc, valA, valB);
                    Value newAcc = b.create<arith::AddFOp>(loc, currAcc, mul);

                    b.create<scf::YieldOp>(loc, newAcc);
                });

            Value dotResult = kLoop.getResult(0);

            // 4. FUSION LOGIC: Add the Bias
            // Load Bias[n] (Broadcasting happens here naturally by reusing nIdx)
            Value valBias = b.create<memref::LoadOp>(loc, biasPtr, ValueRange{nIdx});
            
            // Final = DotResult + Bias
            Value finalRes = b.create<arith::AddFOp>(loc, dotResult, valBias);

            // 5. Store Result -> C[m, n]
            b.create<memref::StoreOp>(loc, finalRes, resPtr, ValueRange{mIdx, nIdx});

            b.create<scf::YieldOp>(loc);
        });
        b.create<scf::YieldOp>(loc);
    });

    // 6. Erase the high-level Op
    rewriter.eraseOp(op);
    return success();
  }
};


struct DotBiasOpToLLVMPass : public mlir::triton::cpu::impl::DotBiasOpToLLVMBase<DotBiasOpToLLVMPass> {
  using mlir::triton::cpu::impl::DotBiasOpToLLVMBase<DotBiasOpToLLVMPass>::DotBiasOpToLLVMBase;

  void runOnOperation() override {

    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    TritonLLVMTypeConverter typeConverter(context);

    // Define what is "Illegal"
    // We want to get rid of all TritonCPU ops
    target.addIllegalDialect<TritonCPUDialect>();
    
    // Define what is "Legal" (LLVM, SCF, Arith, MemRef)
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::memref::MemRefDialect>();

    RewritePatternSet patterns(context);
    
    // Add our new pattern to the list
    patterns.add<DotBiasToLoopsPattern>(typeConverter); // <--- HERE

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
  }
};

}

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createDotBiasOpToLLVMPass() {
  return std::make_unique<DotBiasOpToLLVMPass>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
