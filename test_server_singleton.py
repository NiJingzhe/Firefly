#!/usr/bin/env python3
"""
测试Web服务器的Agent单例行为
"""

import json
import time
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor


def test_model_endpoint():
    """测试模型列表端点"""
    try:
        response = requests.get("http://localhost:8000/v1/models")
        print(f"Models endpoint status: {response.status_code}")
        if response.status_code == 200:
            models = response.json()
            print(f"Available models: {[model['id'] for model in models['data']]}")
        return response.status_code == 200
    except Exception as e:
        print(f"Failed to connect to server: {e}")
        return False


def test_agents_endpoint():
    """测试Agent状态端点"""
    try:
        response = requests.get("http://localhost:8000/v1/agents")
        print(f"Agents endpoint status: {response.status_code}")
        if response.status_code == 200:
            agents_data = response.json()
            print("Registry stats:", agents_data.get("registry_stats", {}))
            print("Active agents:")
            for model_name, agent_info in agents_data.get("agents", {}).items():
                print(f"  {model_name}: ID={agent_info.get('instance_id')}, Name={agent_info.get('name')}")
        return response.status_code == 200
    except Exception as e:
        print(f"Failed to get agents info: {e}")
        return False


def send_chat_request(model_name, message, thread_id=None):
    """发送聊天请求"""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": message}
        ],
        "stream": False
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json=payload,
            timeout=30
        )
        
        thread_info = f" (Thread {thread_id})" if thread_id is not None else ""
        print(f"Chat request{thread_info} - Status: {response.status_code}, Model: {model_name}")
        
        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "model": model_name,
                "response_id": result.get("id"),
                "thread_id": thread_id,
                "content": result["choices"][0]["message"]["content"][:100] + "..." if len(result["choices"][0]["message"]["content"]) > 100 else result["choices"][0]["message"]["content"]
            }
        else:
            return {
                "success": False,
                "model": model_name,
                "error": response.text,
                "thread_id": thread_id
            }
    except Exception as e:
        return {
            "success": False,
            "model": model_name,
            "error": str(e),
            "thread_id": thread_id
        }


def test_singleton_behavior():
    """测试单例行为"""
    print("\n=== 测试Agent单例行为 ===")
    
    # 1. 检查初始状态
    print("\n1. 检查初始Agent状态:")
    test_agents_endpoint()
    
    # 2. 使用同一个模型发送多个请求
    print("\n2. 使用同一个模型发送多个请求:")
    model_name = "simple-agent-v1"
    
    results = []
    for i in range(3):
        result = send_chat_request(model_name, f"测试消息 {i+1}: 你好，请简短回复")
        results.append(result)
        time.sleep(1)  # 避免请求过快
    
    print("\n请求结果:")
    for i, result in enumerate(results):
        print(f"  请求 {i+1}: {'成功' if result['success'] else '失败'}")
        if result['success']:
            print(f"    响应ID: {result['response_id']}")
            print(f"    内容: {result['content']}")
    
    # 3. 检查Agent状态是否一致
    print("\n3. 检查Agent状态:")
    test_agents_endpoint()
    
    # 4. 使用不同模型名称
    print("\n4. 测试不同模型名称:")
    different_model = "test-model-singleton"
    result = send_chat_request(different_model, "你好，这是一个新模型的测试")
    print(f"新模型请求: {'成功' if result['success'] else '失败'}")
    
    # 5. 再次检查Agent状态
    print("\n5. 最终Agent状态:")
    test_agents_endpoint()


def test_concurrent_requests():
    """测试并发请求的单例行为"""
    print("\n=== 测试并发请求的单例行为 ===")
    
    model_name = "concurrent-test-model"
    
    def concurrent_request(thread_id):
        return send_chat_request(model_name, f"并发测试消息 {thread_id}", thread_id)
    
    # 使用线程池发送并发请求
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(concurrent_request, i) for i in range(5)]
        results = [future.result() for future in futures]
    
    print("\n并发请求结果:")
    for result in results:
        print(f"  线程 {result['thread_id']}: {'成功' if result['success'] else '失败'}")
    
    print("\n并发测试后的Agent状态:")
    test_agents_endpoint()


def main():
    """主测试函数"""
    print("🧪 开始测试Web服务器的Agent单例行为")
    
    # 检查服务器是否运行
    if not test_model_endpoint():
        print("❌ 服务器未运行或无法连接，请先启动服务器")
        print("启动命令: python start_web_server.py")
        return
    
    print("✅ 服务器连接成功")
    
    # 运行测试
    test_singleton_behavior()
    test_concurrent_requests()
    
    print("\n✅ 所有测试完成")
    print("\n💡 关键检查点:")
    print("  1. 同一model name的多次请求应该使用相同的Agent实例ID")
    print("  2. 不同model name应该创建不同的Agent实例")
    print("  3. 并发请求应该安全地共享同一个Agent实例")


if __name__ == "__main__":
    main()
