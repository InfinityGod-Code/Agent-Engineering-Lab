import uuid

from a2a.client import create_client
from a2a.types import Message, Part, SendMessageRequest, Role

BASE_URL = "http://localhost:8000"


async def main():
    client = await create_client(agent=BASE_URL)

    message = Message(
        role=Role.ROLE_USER,
        parts=[Part(text="Hello, this is a test message.")],
        message_id=str(uuid.uuid4()),
    )
    request = SendMessageRequest(message=message)

    print("Sending message streaming")
    async for response in client.send_message(request):
        print("Full response:")
        print(response)

    await client.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
