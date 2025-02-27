
import hashlib
import string
import secrets
from datetime import datetime, timedelta
from fastapi import HTTPException
import jwt
import re
import time
import random

from app.models import user
from app import app_config
from app import logger

from app.config.constant import REDIS_USER_REGISTER_CODE, REDIS_USER_LOGIN_CODE, REDIS_USER_RESET_CODE
from app.utils.verification_code_platform import SendSms
from app import redis_client

jwt_config=app_config.jwt_config
SECRET_KEY=jwt_config["jwt_secret_key"]
ALGORITHM = 'HS256'


#------------------------------
#         邀请码部分
#------------------------------

def generate_invitation_code(length=6):
    # 使用 secrets 生成一个包含字母和数字的随机字符串
    characters = string.ascii_letters + string.digits  # 可以根据需要增加字符集
    return ''.join(secrets.choice(characters) for i in range(length))


#------------------------------
#         密钥安全部分
#------------------------------

# 生成 JWT Token
def create_jwt(current_user: user):
    # 设置有效期
    expiration = datetime.now() + timedelta(days=30)  # 七天过期
    payload = {
        "user_id": str(current_user.user_id),
    }
    headers = {"alg": ALGORITHM, "typ": "JWT"}
    # 生成 token
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM, headers=headers)
    return token

def decode_jwt(token: str):
    try:
        # 解码 JWT 并验证签名
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        return payload
    except jwt.ExpiredSignatureError:
        logger.error("Token 已过期!")
        raise HTTPException(status_code=401, detail="Token 已过期")
    except jwt.InvalidTokenError as e:
        logger.error(f"Token 无效! 错误信息: {e}")
        raise HTTPException(status_code=401, detail="Token 无效")
    except Exception as e:
        logger.error(f"解码 JWT 时发生错误: {str(e)}")
        raise HTTPException(status_code=400, detail="无效的 Token")


def md5(s):
    s = s.encode("utf8")
    m = hashlib.md5()
    m.update(s)
    return m.hexdigest()


#------------------------------
#         获取用户部分
#------------------------------

# 获取当前用户信息
def get_current_user(token: str):
    # result = redis_client.get(token)
    # if result == 'expired':
    #     raise HTTPException(status_code=401, detail="Token 已失效")
    # 1. 解码JWT并获取数据
    data = decode_jwt(token)
    logger.debug("data:" + str(data))

    if not data:
        # 如果 token 无效或已过期，返回 401 错误
        raise HTTPException(status_code=401, detail="Token 无效或已过期，请重新登录")

    # 2. 返回当前用户对象
    return data


async def get_code(phone: str, REDIS_PATH: str):
    # 验证手机号格式
    phone_regex = re.compile(r"^1[3-9]\d{9}$")
    if not phone_regex.match(phone):
        logger.error(f"手机号格式不正确！")
        raise HTTPException(status_code=400, detail="手机号格式不正确！")

    # 获取当前时间戳（单位：毫秒）
    timestamp = int(time.time() * 1000)
    # 使用时间戳的一部分与随机数结合
    seed = timestamp + random.randint(0, 9999)
    # 生成随机验证码
    code = (seed % 900000) + 100000
    # 注册
    # SMS_476785298
    # 登录
    # SMS_476855314
    # 重置密码
    # SMS_476695363
    if REDIS_PATH == REDIS_USER_REGISTER_CODE:
        await SendSms.exec(phone, 'SMS_476785298', str(code))
    elif REDIS_PATH == REDIS_USER_LOGIN_CODE:
        await SendSms.exec(phone, 'SMS_476855314', str(code))
    elif REDIS_PATH == REDIS_USER_RESET_CODE:
        await SendSms.exec(phone, 'SMS_476695363', str(code))

    # 存储验证码到Redis中
    result = redis_client.set(REDIS_PATH + phone, str(code), ex=300)  # 过期时间五分钟
    # todo 发送验证码到手机
    logger.info(f"验证码发送到手机号 {phone}: {code}")
    return str(code)


# 验证码校验
async def check_code(code: str, phone: str, REDIS_PATH: str):
    redis_code = redis_client.get(REDIS_PATH + phone)
    logger.info(f"检查代码中ing")
    # 验证码校验
    if code != redis_code:
        raise HTTPException(status_code=400, detail="验证码错误！")
    ttl = redis_client.ttl(REDIS_PATH + phone)
    if ttl <= 0:
        raise HTTPException(status_code=400, detail="验证码已过期！")
    return True

def generate_account():
    """基于时间戳和随机数生成唯一的7位账号"""
    timestamp = int(time.time() * 1000)  # 获取毫秒级时间戳
    random_suffix = random.randint(0, 99)  # 随机生成一个0到99的数字
    account = str(timestamp)[-5:] + f"{random_suffix:02d}"  # 时间戳后5位 + 2位随机数
    return account



