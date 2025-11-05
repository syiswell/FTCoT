import openai


client = openai.OpenAI(
    timeout=60,
    api_key="sk-Pr6Ye2wTLQ05aczR6riYQLjBT4znKzketM93YxfucfYcUZUI",
    base_url="https://pro.xiaoai.plus/v1",
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    temperature=0.0,
)

print(response.choices[0].message.content)
