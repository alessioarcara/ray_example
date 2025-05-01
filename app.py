import ray
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


context = ray.init()
print(context.dashboard_url)


@ray.remote
class MMDetActor:
    def __init__(self, ckpt_path: str):
        self.ckpt_path = ckpt_path

    def predict(self):
        return f"MMDet run with checkpoint {self.ckpt_path}"

    def test_numpy(self):
        import numpy as np
        from loguru import logger
        logger.debug("actor called")
        return {
            "np_version": np.__version__,
            "sum_1_2_3": int(np.array([1,2,3]).sum()),
        }


@ray.remote
class SAMActor:
    def __init__(self):
        pass

    def predict(self):
        from loguru import logger
        logger.debug("actor called")
        return "SAM run"


try:
    sam_actor = ray.get_actor("sam-singleton")
except ValueError:
    sam_actor = SAMActor.options(
        name="sam-singleton",
        namespace="vision",
        lifetime="detached"
    ).remote()


mmdet_actors: dict[str, ray.actor.ActorHandle] = {}


app = FastAPI()


class PredictRequest(BaseModel):
    model: str
    project_name: str | None = None
    checkpoint: str | None = None


@app.post("/predict/")
async def predict(req: PredictRequest):
    if req.model == "mmdet":
        name = req.project_name

        try:
            actor = ray.get_actor(name, namespace="vision")
        except ValueError:
            actor = MMDetActor.options(
                runtime_env={"conda": "/Users/alessioarcara/miniforge3/envs/mmdet"},
                name=name,
                namespace="vision",
                lifetime="detached",
            ).remote(req.checkpoint)

        result = await actor.predict.remote()
        numpy_info = await actor.test_numpy.remote()

        return {
            "model": "mmdet",
            "result": result,
            "numpy_info": numpy_info,
        }

    elif req.model == "sam":
        result = await sam_actor.predict.remote()
        return {"model": "sam", "result": result}

    else:
        raise HTTPException(404, f"Model {req.model} not found")
