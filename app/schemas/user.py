"""
schemas/user.py
用户相关的数据模型
"""

from pydantic import UUID4, EmailStr, constr
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class UserRegister(BaseModel):
    username: Optional[str] = Field(default=None, description="用户的用户名，可选")
    phone: str = Field(..., description="用户的手机号码或者账号必填", min_length=11, max_length=11)
    password: str = Field(..., description="用户的密码，必填", min_length=6, max_length=20)
    verification_code: Optional[str] = Field(default=None, description="验证码，可选", min_length=4, max_length=6)
    invitation_code: Optional[str] = Field(default=None, description="邀请码，可选")

    @field_validator("phone")
    def validate_phone(cls, value):
        if not value.isdigit():
            raise ValueError("手机号码必须为数字")
        return value


class UserLogin(BaseModel):
    phone: str = Field(..., description="用户的手机号码或者账号，必填", min_length=11, max_length=11)
    password: Optional[str] = Field(default=None, description="用户的密码，可选", min_length=6, max_length=20)
    verification_code: Optional[str] = Field(default=None, description="验证码，可选", min_length=4, max_length=6)

    @field_validator("phone")
    def validate_phone(cls, value):
        if not value.isdigit():
            raise ValueError("手机号码必须为数字")
        return value


class UserReset(BaseModel):
    username: Optional[str] = Field(default=None, description="用户的用户名，可选，可以修改")
    phone: str = Field(..., description="用户的手机号码或者账号，必填", min_length=11, max_length=11)
    old_password: Optional[str] = Field(..., description="用户的密码，选填", min_length=6, max_length=20)
    new_password: Optional[str] = Field(..., description="用户的新密码，选填", min_length=6, max_length=20)
    verification_code: Optional[str] = Field(..., description="验证码，选填", min_length=4, max_length=6)
    account: Optional[str] = Field(default=None, description="用户的账号，可选", min_length=11, max_length=11)
    invitation_code: Optional[str] = Field(default=None, description="邀请码，可选")

    @field_validator("phone")
    def validate_phone(cls, value):
        if not value.isdigit():
            raise ValueError("手机号码必须为数字")
        return value
