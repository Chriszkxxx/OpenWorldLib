from typing import List, Optional, Sequence, Union
from PIL import Image as PILImage

from ...reasoning.spatial_reasoning.spatial_reasoner.spatial_reasoner_reasoning import (
    SpatialReasonerReasoning,
)
from ...operators.spatial_reasoner_operator import SpatialReasonerOperator


class SpatialReasonerPipeline:
    """
    Pipeline that builds vision/text inputs and calls SpatialReasonerReasoning; keeps a lightweight operator for interface.
    """

    def __init__(self, reasoning: SpatialReasonerReasoning, operator: SpatialReasonerOperator):
        self.reasoning = reasoning
        self.operator = operator

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "ccvl/SpatialReasoner",
        device: Optional[Union[str, "torch.device"]] = None,
        weight_dtype: "torch.dtype" = None,
        **kwargs,
    ) -> "SpatialReasonerPipeline":
        """
        Args:
            model_path: HuggingFace model ID 或本地模型路径。
            device: 模型加载到的设备（如 "cuda", "cpu"）。
            weight_dtype: 模型权重数据类型（如 torch.bfloat16, torch.float16）。
        """
        import torch
        if weight_dtype is None:
            weight_dtype = torch.bfloat16
        reasoning = SpatialReasonerReasoning.from_pretrained(
            model_path=model_path,
            device=device,
            weight_dtype=weight_dtype,
            **kwargs,
        )
        operator = SpatialReasonerOperator.from_pretrained()
        return cls(reasoning=reasoning, operator=operator)

    def _build_messages(
        self,
        image_inputs: Optional[Union[str, PILImage.Image, Sequence[Union[str, PILImage.Image]]]],
        video_inputs: Optional[Union[str, List[PILImage.Image], Sequence[Union[str, List[PILImage.Image]]]]],
        instruction: str,
    ):
        if image_inputs is None:
            image_inputs = []
        if video_inputs is None:
            video_inputs = []
        # 单个 image：str 或 PIL.Image → 包装成 list
        if isinstance(image_inputs, (str, PILImage.Image)):
            image_inputs = [image_inputs]
        # 单个 video：str → 包装成 list；list[PIL.Image] 视为单个视频帧序列 → 包装成 [frames]
        if isinstance(video_inputs, str):
            video_inputs = [video_inputs]
        elif isinstance(video_inputs, list) and len(video_inputs) > 0 and isinstance(video_inputs[0], PILImage.Image):
            video_inputs = [video_inputs]

        content = [{"type": "image", "image": img} for img in image_inputs]
        content += [{"type": "video", "video": vid} for vid in video_inputs]
        content.append({"type": "text", "text": instruction})
        return [{"role": "user", "content": content}]

    def __call__(
        self,
        instruction: str,
        image_inputs: Optional[Union[str, PILImage.Image, Sequence[Union[str, PILImage.Image]]]] = None,
        video_inputs: Optional[Union[str, List[PILImage.Image], Sequence[Union[str, List[PILImage.Image]]]]] = None,
        max_new_tokens: int = 2048,
        messages: Optional[list] = None,
        generation_kwargs: Optional[dict] = None,
    ) -> List[str]:
        self.operator.get_interaction(instruction)
        self.operator.process_interaction()

        if messages is None:
            batched_messages = [
                self._build_messages(
                    image_inputs=image_inputs,
                    video_inputs=video_inputs,
                    instruction=instruction,
                )
            ]
        else:
            if not messages:
                raise ValueError("messages must be non-empty.")
            batched_messages = [messages] if isinstance(messages[0], dict) else messages

        texts = [
            self.reasoning.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in batched_messages
        ]

        inputs = self.operator.process_perception(batched_messages, texts, processor=self.reasoning.processor)

        outputs = self.reasoning.inference(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            generation_kwargs=generation_kwargs,
        )
        self.operator.delete_last_interaction()
        return outputs
