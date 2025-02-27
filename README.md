# LLM - General

## 简介

包括登录注册，LLM 基本功能的后端项目。

## 主要功能
- 登录注册功能（略）此部分来源于 https://github.com/QodiCat/UserTemplete
- LLM 部分
    - LLM 调用
    - 流式输出
    - 创建历史记录
    - 读取历史记录
    - 删除历史记录
    - 历史记录自动总结

## 使用方法

在 app/config/ 下创建一个 secret.py 文件，格式如下：

```python


#mysql 配置 不变的常量才用大写的
mysql_server=""
mysql_port=
mysql_user=""
mysql_password=""
mysql_database=""


#redis 配置
redis_host=""
redis_password= ""
redis_db=1
redis_pord=6379

#jwt密钥配置
jwt_secrect_config=""


#验证码平台配置
alibaba_cloud_accesskey_iD=""
alibaba_cloud_accesskey_secret= ""
sign_name=""

#大语言模型配置
api_key=""
base_url=""
```

安装 MySQL 和 Redis，将配置文件写入 secret.py，
运行 main.py 自动建表