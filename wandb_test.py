# wandb_test.py
import wandb

# 1) Ensure you’re logged in (will prompt if not)
wandb.login()

# 2) Start a quick run in your actual project/org
run = wandb.init(
    project="RF_Baseline_CNN1D",        # your project name
    name="smoke-test",                  # a memorable run name
    reinit=True                         # allow multiple in one process
)

# 3) Log a dummy metric
wandb.log({"hello": 42})

# 4) Print where it went
print("URL:", run.get_url())

# 5) Close out
run.finish()
