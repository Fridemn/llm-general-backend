import uuid
import os
import edge_tts
import asyncio
from . import TTSProvider

"""
edge_tts 方式，能够免费、快速生成语音，使用需要先安装edge-tts库
```
pip install edge_tts
```
Windows 如果提示找不到指定文件，以管理员身份运行命令行窗口，然后再次运行 AstrBot
"""

class ProviderEdgeTTS(TTSProvider):
    def __init__(
        self,
        provider_config: dict,
        provider_settings: dict,
    ) -> None:
        super().__init__(provider_config, provider_settings)

        # 设置默认语音，如果没有指定则使用中文小萱
        self.voice = provider_config.get("edge-tts-voice", "zh-CN-XiaoxiaoNeural")
        self.rate = provider_config.get("rate", None)
        self.volume = provider_config.get("volume", None)
        self.pitch = provider_config.get("pitch", None)
        self.timeout = provider_config.get("timeout", 30)

        self.set_model("edge_tts")

    async def get_audio(self, text: str) -> str:
        os.makedirs("data/temp", exist_ok=True)
        mp3_path = f"data/temp/edge_tts_temp_{uuid.uuid4()}.mp3"

        # 构建Edge TTS参数
        kwargs = {"text": text, "voice": self.voice}
        if self.rate:
            kwargs["rate"] = self.rate
        if self.volume:
            kwargs["volume"] = self.volume
        if self.pitch:
            kwargs["pitch"] = self.pitch

        try:
            communicate = edge_tts.Communicate(**kwargs)
            await communicate.save(mp3_path)

            if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
                return mp3_path
            else:
                raise RuntimeError("生成的MP3文件不存在或为空")

        except Exception as e:
            try:
                if os.path.exists(mp3_path):
                    os.remove(mp3_path)
            except Exception:
                pass
            raise RuntimeError(f"音频生成失败: {str(e)}")
        

if __name__ == "__main__":
    # 测试
    async def test():
        provider = ProviderEdgeTTS(
            provider_config={
                "id": "edge_tts",
                "type": "edge_tts",
                "edge-tts-voice": "zh-CN-XiaoxiaoNeural",
            },
            provider_settings={},
        )

        audio_path = await provider.get_audio("你好，世界！")
        print(audio_path)

    asyncio.run(test())
