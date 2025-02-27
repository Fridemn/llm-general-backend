
import uuid
from tortoise.models import Model
from tortoise import fields

class User(Model):
    user_id = fields.UUIDField(pk=True, default=uuid.uuid4, index=True)
    phone = fields.CharField(default="", max_length=100, description="电话号")
    account = fields.CharField(default="", max_length=100, description="账户")# 往往和phone相同
    username = fields.CharField(default="", max_length=100, description="用户名")
    password = fields.CharField(default="", max_length=100, description="密码")
    points=fields.IntField(default=0,description="积分")
    invitation_code=fields.CharField(default="", max_length=100, description="邀请码")

    



