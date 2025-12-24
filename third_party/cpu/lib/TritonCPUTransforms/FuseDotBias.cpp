#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h" // tt dialect
#include "triton/Dialect/TritonCPU/IR/Dialect.h" // Your target dialect
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// Include the generated declarations which define the base class.
// According to Passes.h, this should be included with GEN_PASS_DEF_FUSEDOTBIAS
namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_FUSEDOTBIAS
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

struct FuseDotBiasPattern : public OpRewritePattern<arith::AddFOp> {
 // Constructor: Initialize with the context and benefit (priority)
 FuseDotBiasPattern(MLIRContext *context)
     : OpRewritePattern<arith::AddFOp>(context, /*benefit=*/1) {}


 // The Core Logic
 LogicalResult matchAndRewrite(arith::AddFOp addOp,
                               PatternRewriter &rewriter) const override {
  
   // Step 1: Get the two operands of the Add operation
   Value lhs = addOp.getLhs();
   Value rhs = addOp.getRhs();


   // Step 2: Identify which operand is the Dot and which is the Bias
   // We use dyn_cast to check if the Value is defined by a specific Op
   triton::DotOp dotOp = lhs.getDefiningOp<triton::DotOp>();
   Value bias = rhs;


   if (!dotOp) {
     // Try the other way around (Add is commutative)
     dotOp = rhs.getDefiningOp<triton::DotOp>();
     bias = lhs;
   }


   // If neither operand is a DotOp, this pattern doesn't apply. Fail.
   if (!dotOp) {
     return failure();
   }


   // Step 3: Safety Checks (Crucial for compilers)
   // We can only fuse if the DotOp has NO other users.
   // If 'dotOp' is used elsewhere, fusing it would break that other usage.
   if (!dotOp.getResult().hasOneUse()) {
     return failure();
   }


   // Step 4: Create the New Fused Operation
   // We assume you have defined 'triton_cpu.dot_bias' in your TableGen.
   // Inputs: Dot LHS (A), Dot RHS (B), Bias (C)
  
   // Set the insertion point to where the AddOp currently is
   rewriter.setInsertionPoint(addOp);

   // Retrieve attributes from the original DotOp to preserve precision settings
   auto inputPrecision = dotOp.getInputPrecisionAttr();
   auto maxNumImpreciseAcc = dotOp.getMaxNumImpreciseAccAttr();

   // Fix: DotBiasOp is in mlir::triton::cpu namespace (aliased by using namespace above)
   auto fusedOp = rewriter.create<DotBiasOp>(
       addOp.getLoc(),       // Source location for debug info
       addOp.getType(),      // Result type (same as the original Add)
       dotOp.getA(),         // Operand A from old Dot
       dotOp.getB(),         // Operand B from old Dot
       bias,                  // The Bias operand
       inputPrecision,       // <--- ADDED: Preserve precision (TF32 vs F32)
       maxNumImpreciseAcc    // <--- ADDED: Preserve accumulation mode
   );


   // Step 5: Replace the Old Op with the New Op
   // This tells MLIR: "Wherever 'addOp' was used, use 'fusedOp' now."
   rewriter.replaceOp(addOp, fusedOp.getResult());


   // Note: We don't manually delete 'dotOp'.
   // MLIR's Dead Code Elimination (DCE) pass will clean it up later
   // because it now has zero users.


   return success();
 }
};

// Fix: Address the ambiguity by fully qualifying the base class
struct FuseDotBias : public mlir::triton::cpu::impl::FuseDotBiasBase<FuseDotBias> {

    void runOnOperation() override {
        MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);
        // Add our pattern to the list
        patterns.add<FuseDotBiasPattern>(context);
        // Run the greedy pattern rewriter
        // It will apply the pattern repeatedly until convergence
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
   }

};


} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createFuseDotBias() {
  return std::make_unique<FuseDotBias>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
