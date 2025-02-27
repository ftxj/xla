#!/bin/bash
set -exo pipefail
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
LOGFILE=/tmp/pytorch_py_test.log
MAX_GRAPH_SIZE=500
GRAPH_CHECK_FREQUENCY=100
VERBOSITY=2

while getopts 'LM:C:V:' OPTION
do
  case $OPTION in
    L)
      LOGFILE=
      ;;
    M)
      MAX_GRAPH_SIZE=$OPTARG
      ;;
    C)
      GRAPH_CHECK_FREQUENCY=$OPTARG
      ;;
    V)
      VERBOSITY=$OPTARG
      ;;
  esac
done
shift $(($OPTIND - 1))

export TRIM_GRAPH_SIZE=$MAX_GRAPH_SIZE
export TRIM_GRAPH_CHECK_FREQUENCY=$GRAPH_CHECK_FREQUENCY
export TORCH_TEST_DEVICES="$CDIR/pytorch_test_base.py"
export PYTORCH_TEST_WITH_SLOW=1
export XLA_DUMP_FATAL_STACK=1
export CPU_NUM_DEVICES=4

function run_test {
  "$@"
}

function run_opbyop {
  echo "Running in OpByOp mode: $@"
  XLA_GET_TENSORS_OPBYOP=1 XLA_SYNC_TENSORS_OPBYOP=1 run_test "$@"
}

function run_use_bf16 {
  echo "Running with XLA_USE_BF16: $@"
  XLA_USE_BF16=1 run_test "$@"
}

function run_downcast_bf16 {
  echo "Running with XLA_DOWNCAST_BF16: $@"
  XLA_DOWNCAST_BF16=1 run_test "$@"
}

function run_xla_ir_debug {
  echo "Running with XLA_IR_DEBUG: $@"
  XLA_IR_DEBUG=1 run_test "$@"
}

function run_dynamic {
  if [[ "$TPUVM_MODE" == "1" ]]; then
    run_test "$@"
  else
    echo "Running in DynamicShape mode: $@"
    XLA_EXPERIMENTAL="nonzero:masked_select:masked_scatter" run_test "$@"
  fi
}

function run_eager_debug {
  echo "Running in Eager Debug mode: $@"
  XLA_USE_EAGER_DEBUG_MODE=1 run_test "$@"
}

function run_xla_backend_mp {
  echo "Running XLA backend multiprocessing test: $@"
  MASTER_ADDR=localhost MASTER_PORT=6000 run_test "$@"
}

function run_async_rng {
  echo "Running in Async RNG Upload mode: $@"
  XLA_TRANSFER_SEED_ASYNC=1 run_test "$@"
}

function run_all_tests {
  run_dynamic python3 "$CDIR/../../test/test_view_ops.py" "$@" -v TestViewOpsXLA
  run_test python3 "$CDIR/../../test/test_torch.py" "$@" -v TestTorchDeviceTypeXLA
  run_dynamic python3 "$CDIR/../../test/test_torch.py" "$@" -v TestDevicePrecisionXLA
  run_test python3 "$CDIR/../../test/test_torch.py" "$@" -v TestTensorDeviceOpsXLA
  run_test python3 "$CDIR/../../test/test_indexing.py" "$@" -v TestIndexingXLA
  run_test python3 "$CDIR/../../test/test_indexing.py" "$@" -v NumpyTestsXLA
  run_dynamic python3 "$CDIR/../../test/test_nn.py" "$@" -v TestNNDeviceTypeXLA
  run_dynamic python3 "$CDIR/../../test/test_type_promotion.py" "$@" -v TestTypePromotionXLA
  run_dynamic python3 "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  run_opbyop python3 "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  run_eager_debug python3 "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  run_async_rng python3 "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  # TODO: enable this test after tf update, currently optimization_barrier does not
  # work on CPU.
  # run_test python3 "$CDIR/test_checkpoint.py"
  run_test python3 "$CDIR/test_mp_replication.py"
  run_test python3 "$CDIR/test_mp_all_to_all.py"
  run_test python3 "$CDIR/test_mp_collective_permute.py"
  run_test python3 "$CDIR/test_mp_all_gather.py"
  run_test python3 "$CDIR/test_mp_reduce_scatter.py"
  run_test python3 "$CDIR/test_mp_distributed_mm.py"
  run_test python3 "$CDIR/test_mp_rendezvous.py"
  run_test python3 "$CDIR/test_mp_save.py"
  run_test python3 "$CDIR/test_mp_mesh_reduce.py"
  run_test python3 "$CDIR/test_mp_sync_batch_norm.py"
  run_test python3 "$CDIR/test_async_closures.py"
  run_test python3 "$CDIR/test_xla_dist.py"
  run_test python3 "$CDIR/test_profiler.py"
  run_test python3 "$CDIR/test_ops.py"
  run_downcast_bf16 python3 "$CDIR/test_data_type.py"
  run_use_bf16 python3 "$CDIR/test_data_type.py"
  run_test python3 "$CDIR/test_torch_distributed_xla_backend.py"
  run_xla_backend_mp python3 "$CDIR/test_torch_distributed_all_gather_xla_backend.py"
  run_xla_backend_mp python3 "$CDIR/test_torch_distributed_all_reduce_xla_backend.py"
  run_xla_backend_mp python3 "$CDIR/test_torch_distributed_multi_all_reduce_xla_backend.py"
  run_xla_backend_mp python3 "$CDIR/test_torch_distributed_reduce_scatter_xla_backend.py"
  run_xla_ir_debug python3 "$CDIR/test_env_var_mapper.py"
}

if [ "$LOGFILE" != "" ]; then
  run_all_tests 2>&1 | tee $LOGFILE
else
  run_all_tests
fi
