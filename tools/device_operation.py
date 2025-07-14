import asyncio
import websockets
from websockets import Data
import uuid
import time
import json
from typing import Any, Dict
from SimpleLLMFunc import tool
from .common import safe_asyncio_run, print_tool_output


class WebSocketClient:
    def __init__(self, endpoint: str) -> None:
        self.endpoint: str = endpoint
        self.session: str = str(uuid.uuid4())

    async def send_message(
        self, action: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        msg: Dict[str, Any] = {
            "session": self.session,
            "msgid": str(uuid.uuid4()),
            "timestamp": int(time.time()),
            "action": action,
            "payload": payload,
        }
        async with websockets.connect(self.endpoint) as ws:
            await ws.send(json.dumps(msg))
            response: Data = await ws.recv()
            return json.loads(response)

    async def get_device_list(self) -> Dict[str, Any]:
        return await self.send_message("list_device", {})

    async def get_value(self, payload_json_str: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = json.loads(payload_json_str)
        return await self.send_message("get_value", payload)

    async def set_value(self, payload_json_str: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = json.loads(payload_json_str)
        return await self.send_message("set_value", payload)


@tool(
    name="get_device_list",
    description="得到所有可控制的设备列表，包含设备的id，类型以及可控制参数等信息，还会返回操作工具需要的schema要求",
)
def get_device_list() -> str:
    """
    取得相关智能设备的参数和名称说明的列表

    Args:

    Return:
        Dict[str, Any]: 设备列表,{"device": [device1, device2, ...]}
    """
    device_list: list[dict[str, str]] = safe_asyncio_run(client.get_device_list())[
        "devices"
    ]

    # 根据device list推导应该有的get 和 set value 的 schema
    did = []
    dtype = []
    dparameters = []
    for device in device_list:
        did.append(device["id"])
        dtype.append(device["type"])
        dparameters.append(device["parameters"])

    get_schema: list[str] = []
    set_schema: list[str] = []

    for i, id in enumerate(did):
        get_schema.append(
            "{\n"
            + '"id": '
            + str(id)
            + "],\n"
            + '"parameters": [\n'
            + ", ".join([str(param_name) for param_name in dparameters[i]])
            + "]\n"
            + "}"
        )

        set_schema.append(
            "{\n"
            + '"id": '
            + str(id)
            + "],\n"
            + '"parameters": {\n'
            + ",\n".join(dparameters[i])
            + "\n"
            + "}\n"
            + "}"
        )

    print_tool_output(title="获取到智能设备列表", content=str(device_list))

    print_tool_output(title="获取到get_device_value的schema", content=str(get_schema))

    print_tool_output(title="获取到set_device_value的schema", content=str(set_schema))

    # 返回内容
    return_str = (
        "得到了所有设备的列表如下："
        f"{device_list}\n"
        "同时还有对应的get_device_value和set_device_value的schema如下：\n"
        f"get_schema:\n {get_schema}\n\n"
        f"set_schema:\n {set_schema}\n\n"
        "后续设备操作请严格遵循以上要求的schema，尤其注意id和parameters的对应，以及parameters的类型以及范围要求。"
    )

    return return_str


@tool(
    name="get_device_value",
    description="获取设备的值，返回设备的当前状态和参数等信息",
)
def get_device_value(payload_json_str: str) -> str:
    """
    Args:
        payload_json_str (str): 设备id和参数的json字符串, 例如：{"id": "abc123", "parameters": ["temperature", "humidity"]}, 请严格根据get device list返回的参考schema填写。
    Return:
        str: 设备的当前状态和参数等信息
    """
    result: Dict[str, Any] = safe_asyncio_run(client.get_value(payload_json_str))

    print_tool_output(
        title="获取到设备的值",
        content=str(result),
    )

    return_str: str = (
        "获取到设备的值如下：\n"
        f"{result}\n"
        "请注意，返回的值可能包含多个参数，请根据实际需要进行处理。"
    )

    return return_str


@tool(
    name="set_device_value",
    description="设置设备的值",
)
def set_device_value(payload_json_str: str) -> str:
    """
    Args:
        payload_json_str (str): 设备id和参数的json字符串, 例如：{"id": "abc123", "parameters": {"temperature": 22, "humidity": 50}}, 请严格根据get device list返回的参考schema填写。
    Return:
        str: 设置设备的结果
    """
    result: Dict[str, Any] = safe_asyncio_run(client.set_value(payload_json_str))

    print_tool_output(
        title="设置设备的值",
        content=str(result),
    )

    resutn_str: str = (
        "设置设备的值如下：\n"
        f"{result}\n"
        "请注意，返回的结果可能包含操作是否成功的信息，请根据实际需要进行处理。"
    )

    return resutn_str


if __name__ == "__main__":
    endpoint: str = "ws://normalalkene.dynv6.net:13579"
    client: WebSocketClient = WebSocketClient(endpoint)

    async def main() -> None:
        device_list: Dict[str, Any] = await client.get_device_list()
        print("Device List:", device_list)

        get_payload: str = '{"device_id": "abc123"}'
        value: Dict[str, Any] = await client.get_value(get_payload)
        print("Get Value:", value)

        set_payload: str = '{"device_id": "abc123", "value": 42}'
        result: Dict[str, Any] = await client.set_value(set_payload)
        print("Set Value:", result)

    print("start")

    asyncio.run(main())

    print("end")
