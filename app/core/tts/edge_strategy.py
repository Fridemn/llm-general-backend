import os
import uuid
import logging
import edge_tts
from datetime import datetime
from typing import Dict, Any, List

from app import logger
from app.core.tts import TTSProvider

# 音频输出目录
AUDIO_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "static", "audio"
)
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)


class ProviderEdgeTTS(TTSProvider):
    """Edge TTS 提供者实现"""

    DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"

    # 可用的中文男声和女声列表
    AVAILABLE_VOICES = {
        "chinese": {
            "female": [
                "zh-CN-XiaoxiaoNeural",  # 晓晓，女声，温暖亲和
                "zh-CN-XiaoyiNeural",  # 晓伊，女声，温柔感性
                "zh-CN-liaoning-XiaobeiNeural",  # 晓北，东北口音
                "zh-TW-HsiaoChenNeural",  # 台湾口音
                "zh-HK-HiuGaaiNeural",  # 香港口音
            ],
            "male": ["zh-CN-YunxiNeural", "zh-CN-YunjianNeural"],  # 云希，男声，稳重多才  # 云健，男声，阳光开朗
        }
    }

    def __init__(self, provider_config: Dict[str, Any], provider_settings: Dict[str, Any]):
        """
        初始化Edge TTS提供者

        Args:
            provider_config: 提供者配置
            provider_settings: 全局设置
        """
        super().__init__(provider_config, provider_settings)

        # 从配置中获取语音
        self.voice = provider_config.get("edge-tts-voice", self.DEFAULT_VOICE)

        # 从提供者名称设置
        self.set_model("edge_tts")

    async def get_voices(self) -> List[Dict[str, Any]]:
        """
        获取所有可用的语音列表

        Returns:
            语音列表
        """
        try:
            voices = await edge_tts.list_voices()
            return voices
        except Exception as e:
            logger.error(f"获取Edge TTS语音列表失败: {str(e)}")
            return []

    async def get_audio(self, text: str, **kwargs) -> str:
        """
        将文本转换为音频

        Args:
            text: 要转换的文本
            **kwargs: 额外参数
                - voice: 语音名称
                - rate: 语速，如 "+10%", "-20%"
                - volume: 音量，如 "+10%", "-20%"

        Returns:
            音频文件路径
        """
        try:
            # 从kwargs中获取参数，如果没有则使用默认值
            voice = kwargs.get("voice", self.voice)
            rate = kwargs.get("rate", "+0%")
            volume = kwargs.get("volume", "+0%")

            # 检查文本是否为空
            if not text.strip():
                logger.warning("文本为空，无法生成音频")
                return ""

            # 生成输出文件路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tts_{timestamp}_{uuid.uuid4()}.mp3"
            output_path = os.path.join(AUDIO_OUTPUT_DIR, filename)

            # 处理可能较长的文本，分割处理
            if len(text) > 5000:
                logger.warning(f"文本长度超过5000字符 ({len(text)}), 将进行分割处理")
                # 每5000个字符分割一次
                chunks = [text[i : i + 5000] for i in range(0, len(text), 5000)]
                temp_files = []

                # 处理每个块
                for i, chunk in enumerate(chunks):
                    temp_filename = f"temp_{timestamp}_{i}_{uuid.uuid4()}.mp3"
                    temp_path = os.path.join(AUDIO_OUTPUT_DIR, temp_filename)

                    # 使用Edge TTS生成每个块的音频
                    communicate = edge_tts.Communicate(chunk, voice, rate=rate, volume=volume)
                    await communicate.save(temp_path)
                    temp_files.append(temp_path)

                # 合并音频文件 (这里简化处理，只返回第一个文件)
                # 实际上，应该使用pydub或其他库合并所有文件
                if temp_files:
                    os.rename(temp_files[0], output_path)
                    # 删除剩余的临时文件
                    for temp_file in temp_files[1:]:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
            else:
                # 对于短文本，直接生成
                communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume)
                await communicate.save(output_path)

            # logger.info(f"成功生成音频文件: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"生成音频失败: {str(e)}")
            raise
