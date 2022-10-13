import os
import neptune.new as neptune

if 'NEPTUNE_TOKEN' in os.environ and 'NEPTUNE_PROJECT' in os.environ:
    run = neptune.init(
        project=os.environ['NEPTUNE_PROJECT'],
        api_token=os.environ['NEPTUNE_TOKEN'],
    )  # your credentials
    NEPTUNE = True
else:
    NEPTUNE = False

params = {"learning_rate": 0.001, "optimizer": "Adam"}
if NEPTUNE:
    run["parameters"] = params

for epoch in range(100):
    pass
    if NEPTUNE:
        run["train/loss"].log(0.87 ** epoch)


if NEPTUNE:
    run["eval/f1_score"] = 0.66
    run.stop()
print('finished')
