import asyncio

from gs_video.api.middleware import (
    StorageMaintenanceBoundary,
    StorageMaintenanceCoordinator,
)
from gs_video.api.schemas import ApiSettings


def test_exclusive_maintenance_waits_for_mutations_and_blocks_new_ones() -> None:
    async def exercise() -> None:
        coordinator = StorageMaintenanceCoordinator()
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        exclusive_entered = asyncio.Event()
        release_exclusive = asyncio.Event()
        second_entered = asyncio.Event()

        async def first_mutation() -> None:
            async with coordinator.mutation():
                first_entered.set()
                await release_first.wait()

        async def exclusive() -> None:
            async with coordinator.exclusive():
                exclusive_entered.set()
                await release_exclusive.wait()

        async def second_mutation() -> None:
            async with coordinator.mutation():
                second_entered.set()

        first = asyncio.create_task(first_mutation())
        await first_entered.wait()
        maintenance = asyncio.create_task(exclusive())
        await asyncio.sleep(0)
        assert not exclusive_entered.is_set()
        release_first.set()
        await exclusive_entered.wait()
        second = asyncio.create_task(second_mutation())
        await asyncio.sleep(0)
        assert not second_entered.is_set()
        release_exclusive.set()
        await asyncio.gather(first, maintenance, second)
        assert second_entered.is_set()

    asyncio.run(exercise())


def test_waiting_mutation_rechecks_restart_requirement_after_admission() -> None:
    async def exercise() -> None:
        coordinator = StorageMaintenanceCoordinator()
        blocked = False
        dispatched = False
        messages: list[dict[str, object]] = []

        async def app(_scope, _receive, _send) -> None:  # type: ignore[no-untyped-def]
            nonlocal dispatched
            dispatched = True

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, object]) -> None:
            messages.append(message)

        boundary = StorageMaintenanceBoundary(
            app,
            coordinator=coordinator,
            settings=ApiSettings(
                bind_host="127.0.0.1",
                port=0,
                session_token="test-token",
                allowed_origins=(),
            ),
            mutation_blocked=lambda: blocked,
        )
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/assets/import",
            "headers": [],
        }
        async with coordinator.exclusive():
            request = asyncio.create_task(boundary(scope, receive, send))  # type: ignore[arg-type]
            await asyncio.sleep(0)
            blocked = True
        await request

        assert dispatched is False
        assert messages[0]["status"] == 409

    asyncio.run(exercise())
