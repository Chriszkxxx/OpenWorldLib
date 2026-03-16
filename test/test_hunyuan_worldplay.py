import os
from openworldlib.pipelines.hunyuan_world.pipeline_hunyuan_worldplay import HunyuanWorldPlayPipeline

image_path = "./data/test_case/test_image_case1/ref_image.png"
prompt = "A cozy snowy fairy-tale village with thatched cottages covered in thick snow."
interaction_signal = ["forward", "camera_l", "camera_r"]
video_sync_path = "tencent/HunyuanVideo-1.5"
action_ckpt = "tencent/HY-WorldPlay"

output_path = "./outputs"
os.makedirs(output_path, exist_ok=True)

pipeline = HunyuanWorldPlayPipeline.from_pretrained(
    model_path=action_ckpt,
    mode="480p_i2v",
    required_components = {"video_model_path": video_sync_path},
    enable_offloading=True,
    device="cuda"
)
output = pipeline(
    prompt=prompt,
    image_path=image_path,
    interactions=interaction_signal,
)

save_video_path = os.path.join(output_path, "hunyuan_worldplay_demo.mp4")
HunyuanWorldPlayPipeline.save_video(output.videos, save_video_path)
