"""
user.py
存放用户相关接口，如登录注册修改信息等
"""

from fastapi import APIRouter, HTTPException, Depends
from tortoise.exceptions import IntegrityError
from tortoise.expressions import Q

from app.config.constant import *
from app import redis_client, app_config, logger
from app.models.user import User
from app.schemas.response import ResponseModel
from app.schemas.user import UserRegister, UserLogin, UserReset
from app.utils.user import create_jwt, get_current_user, get_code, check_code, generate_account, md5


def get_user_by_phone(phone):
    return User.filter(phone=phone).first()


api_user = APIRouter()


# ------------------------------
# 用户注册部分,有两种注册方式，一种是手机号+密码，一种是手机号+验证码
# ------------------------------


@api_user.get("/register/verification-code/send", description="发送注册验证码，必填手机号")
async def register_verification_code_get(phone: str):
    REDIS_USER_REGISTER_CODE = app_config.redis_config["user_register_code"]
    await get_code(phone, REDIS_USER_REGISTER_CODE)
    return ResponseModel.success("注册账号验证码发送成功")


@api_user.post(
    "/register/verification-code-way", description="通过验证码的方式进行用户注册，验证刚刚发送的注册验证码，邀请码选填"
)
async def register_by_verification_code(user_register: UserRegister):
    username = user_register.username
    phone = user_register.phone
    password = user_register.password
    verification_code = user_register.verification_code
    invitation_code = user_register.invitation_code

    user_config = app_config["user_config"]
    init_points = user_config["user_points"]["init_points"]
    add_points = user_config["user_points"]["invite_points"]

    if not username:
        username = "用户" + str(phone)
    if verification_code:
        await check_code(verification_code, phone, REDIS_USER_REGISTER_CODE)
    user = await User.filter(phone=phone).first()

    if user:
        token = create_jwt(user)
        return ResponseModel.success("用户已存在，已登录", {"current_user": get_current_user(token), "token": token})
    else:
        # 用户不存在，创建新用户并直接登录,两者邀请加积分
        inviting_user = await User.filter(invitation_code=invitation_code).first()
        if inviting_user:
            inviting_user_old_points = inviting_user.points
            inviting_user.points = add_points + inviting_user_old_points
            await inviting_user.save()
            new_user_points = add_points + init_points
        # 新建用户
        try:
            new_user_points = init_points
            newUser = await User.create(
                account=phone, phone=phone, username=username, password=md5(password), points=new_user_points
            )
            token = create_jwt(newUser)
            return ResponseModel.success("注册用户成功", {"current_user": get_current_user(token), "token": token})
        except IntegrityError:
            raise HTTPException(status_code=500, detail="用户创建失败，请稍后重试")


@api_user.post("/register/password-way", description="通过密码的方式进行用户注册，邀请码选填")
async def register_by_password(user_register: UserRegister):
    username = user_register.username
    phone = user_register.phone
    password = user_register.password
    invitation_code = user_register.invitation_code

    user_config = app_config["user_config"]
    init_points = user_config["user_points"]["init_points"]
    add_points = user_config["user_points"]["invite_points"]

    if not username:
        username = "用户" + str(phone)
    user = await User.filter(phone=phone).first()
    if user:
        token = create_jwt(user)
        return ResponseModel.success("用户已存在，已登录", {"current_user": get_current_user(token), "token": token})
    else:
        # 用户不存在，创建新用户并直接登录,两者邀请加积分
        inviting_user = await User.filter(invitation_code=invitation_code).first()
        if inviting_user:
            inviting_user_old_points = inviting_user.points
            inviting_user.points = add_points + inviting_user_old_points
            await inviting_user.save()
        new_user_points = add_points + init_points
        # 新建用户
        try:
            newUser = await User.create(
                account=phone, phone=phone, username=username, password=md5(password), points=new_user_points
            )
            token = create_jwt(newUser)
            return ResponseModel.success("注册用户成功", {"current_user": get_current_user(token), "token": token})
        except IntegrityError:
            raise HTTPException(status_code=500, detail="用户创建失败，请稍后重试")


# ------------------------------
# 用户登录部分,有两种登录方式，一种是手机号+密码，一种是手机号+验证码
# ------------------------------


@api_user.get("/login/verification-code/send", description="发送登录验证码")
async def login_verification_code_get(phone: str):
    REDIS_USER_LOGIN_CODE = app_config.redis_config["user_login_code"]
    await get_code(phone, REDIS_USER_LOGIN_CODE)
    return ResponseModel.success("登录账号验证码发送成功")


@api_user.post(
    "/login/verification-code-way", description="通过验证码的方式进行用户登录，此时必须输入验证码和电话，密码是null"
)
async def login_by_verification_code(user_login: UserLogin):
    phone = user_login.phone
    code = user_login.code
    if code == None:
        raise HTTPException(status_code=400, detail="请输入验证码")
    REDIS_USER_LOGIN_CODE = app_config.redis_config["user_login_code"]
    await check_code(code, phone, REDIS_USER_LOGIN_CODE)

    user = await User.filter(phone=phone).first()
    if user:
        token = create_jwt(user)
        return ResponseModel.success("登录成功", {"current_user": get_current_user(token), "token": token})
    else:
        raise HTTPException(status_code=404, detail="用户不存在")


@api_user.post("/login/password-way", description="通过密码的方式进行用户登录，此时必须输入密码和电话，验证码是null")
async def login_by_password(user_login: UserLogin):
    phone = user_login.phone
    password = user_login.password
    if password == None:
        raise HTTPException(status_code=400, detail="请输入密码")

    user = await User.filter(phone=phone).first()
    if user:
        if user.password == md5(password):
            token = create_jwt(user)
            return ResponseModel.success("登录成功", {"current_user": get_current_user(token), "token": token})
        else:
            raise HTTPException(status_code=401, detail="密码错误")
    else:
        raise HTTPException(status_code=404, detail="用户不存在")


@api_user.post("/logout", description="退出登录")
async def logout(token: str):
    redis_client.set(token, "expired", ex=60 * 60 * 24)
    return ResponseModel.success("退出登录成功")


# ------------------------------
# 用户修改部分,修改密码，修改其他信息
# ------------------------------


@api_user.get("/reset/verification-code/send", description="发送重置验证码")
async def reset_info_verification_code_get(phone: str):
    REDIS_USER_RESET_CODE = app_config.redis_config["user_reset_code"]
    await get_code(phone, REDIS_USER_RESET_CODE)
    return ResponseModel.success("重置验证码发送成功")


@api_user.post(
    "/reset/info/verification-code-way",
    description="通过验证码的方式重置信息，此时必须输入验证码和电话，其他选择性填充即可",
)
async def reset_info_by_verification_code_way(user_reset: UserReset):
    phone = user_reset.phone
    new_password = user_reset.new_password
    verification_code = user_reset.verification_code
    invitation_code = user_reset.invitation_code
    username = user_reset.username
    account = user_reset.account

    if verification_code == None:
        raise HTTPException(status_code=400, detail="请输入验证码")

    REDIS_USER_RESET_CODE = app_config.redis_config["user_reset_code"]
    await check_code(verification_code, phone, REDIS_USER_RESET_CODE)

    user = await User.filter(phone=phone).first()
    if user:
        if new_password:
            user.password = md5(new_password)
        if username:
            user.username = username
        if invitation_code:
            user.invitation_code = invitation_code
        if phone:
            user.phone = phone
        if account:
            user.account = account
        await user.save()
        return ResponseModel.success("重置信息成功")
    else:
        raise HTTPException(status_code=404, detail="用户不存在")


@api_user.post(
    "/reset/info/password-way", description="通过密码的方式重置信息，此时必须输入原密码和电话，其他选择性填充即可"
)
async def reset_info_by_password(user_reset: UserReset):
    phone = user_reset.phone
    old_password = user_reset.old_password
    if not old_password:
        raise HTTPException(status_code=400, detail="请输入原本密码")
    user = await User.filter(phone=phone).first()
    if user:
        if user.password != md5(old_password):
            raise HTTPException(status_code=401, detail="密码错误")
        else:
            new_password = user_reset.new_password
            invitation_code = user_reset.invitation_code
            username = user_reset.username
            account = user_reset.account
            if new_password:
                user.password = md5(new_password)
            if username:
                user.username = username
            if invitation_code:
                user.invitation_code = invitation_code
            if phone:
                user.phone = phone
            if account:
                user.account = account
            await user.save()
            return ResponseModel.success("重置信息成功")
    else:
        raise HTTPException(status_code=404, detail="用户不存在")


# ------------------------------
# 用户查看部分
# ------------------------------
@api_user.get("/profile", description="获取当前用户信息")
async def get_profile(token: str):
    current_user = get_current_user(token)
    return ResponseModel.success("获取用户信息成功", current_user)


@api_user.get("/all-users", description="查询全部用户")
async def get_all_users(token: str):
    current_user = get_current_user(token)
    user_list = await User.filter().all()
    return user_list
