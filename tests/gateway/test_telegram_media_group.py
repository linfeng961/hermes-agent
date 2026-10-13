"""Album (media-group) merge: a slow/sequential per-photo download must NOT split a Telegram
album so the model sees only the first image.

Regressions for "I sent N photos as one album, the model saw 1 image + my caption". Two
mechanisms keep the album whole:
  1. The debounce starts on item ARRIVAL (before the photo download round-trip), so a slow
     sibling still resets the window.
  2. The flush holds (bounded by MEDIA_GROUP_MAX_WAIT_SECONDS) while sibling downloads are
     still in flight, and every item exit path releases its in-flight slot.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.event import MessageEvent, MessageType


def _adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    a = object.__new__(TelegramAdapter)
    a._media_group_events = {}
    a._media_group_tasks = {}
    a.MEDIA_GROUP_WAIT_SECONDS = 0.2
    a.MEDIA_GROUP_MAX_WAIT_SECONDS = 1.0
    a.handle_message = AsyncMock()
    return a


def _photo_event(url, text=""):
    e = MessageEvent(text=text, message_type=MessageType.PHOTO)
    e.media_urls = [url]
    e.media_types = ["image/jpeg"]
    return e


def _dispatched_urls(a):
    return [list(call.args[0].media_urls) for call in a.handle_message.await_args_list]


@pytest.mark.asyncio
async def test_album_basic_two_photos_merge_into_one_event():
    a = _adapter()

    a._note_media_group_arrival("g")
    a._note_media_group_arrival("g")
    await a._queue_media_group_event("g", _photo_event("p1", "caption A"))
    a._media_group_item_finished("g")
    await a._queue_media_group_event("g", _photo_event("p2"))
    a._media_group_item_finished("g")
    await asyncio.sleep(0.5)

    assert a.handle_message.await_count == 1, f"album split into {a.handle_message.await_count} messages"
    assert _dispatched_urls(a) == [["p1", "p2"]]


@pytest.mark.asyncio
async def test_album_survives_slow_sibling_download():
    """The 2nd photo finishes downloading AFTER the debounce window — the album must still
    arrive whole (the exact failure mode of the original bug)."""
    a = _adapter()

    # Both album updates arrive ~together (before their downloads).
    a._note_media_group_arrival("g")
    a._note_media_group_arrival("g")
    # Item 1 downloads fast and is buffered.
    await a._queue_media_group_event("g", _photo_event("p1", "caption A"))
    a._media_group_item_finished("g")
    # Item 2's download is SLOW — longer than MEDIA_GROUP_WAIT_SECONDS.
    await asyncio.sleep(0.4)
    await a._queue_media_group_event("g", _photo_event("p2"))
    a._media_group_item_finished("g")
    await asyncio.sleep(0.5)

    assert a.handle_message.await_count == 1, f"slow sibling split the album into {a.handle_message.await_count} messages"
    assert _dispatched_urls(a) == [["p1", "p2"]]


@pytest.mark.asyncio
async def test_album_five_photos_all_reach_model():
    a = _adapter()

    for _ in range(5):
        a._note_media_group_arrival("g")
    for i in range(5):
        await a._queue_media_group_event("g", _photo_event(f"p{i}"))
        a._media_group_item_finished("g")
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.5)

    assert a.handle_message.await_count == 1
    assert _dispatched_urls(a) == [[f"p{i}" for i in range(5)]]


@pytest.mark.asyncio
async def test_album_flush_cap_releases_wedged_sibling():
    """One wedged download must not hold the album forever: the flush fires anyway after
    MEDIA_GROUP_MAX_WAIT_SECONDS with whatever it has."""
    a = _adapter()
    a.MEDIA_GROUP_MAX_WAIT_SECONDS = 0.5

    a._note_media_group_arrival("g")
    await a._queue_media_group_event("g", _photo_event("p1"))
    a._note_media_group_arrival("g")  # sibling arrives but its download never finishes

    await asyncio.sleep(0.9)  # 0.2 delay + 0.5 cap + margin

    assert a.handle_message.await_count == 1, "wedged sibling wedged the whole album"
    assert _dispatched_urls(a) == [["p1"]]


@pytest.mark.asyncio
async def test_inflight_slot_released_on_early_return_without_queueing():
    """An item that leaves the pipeline WITHOUT queueing (drop / oversized / held) must still
    release its in-flight slot — otherwise the flush waits out the full cap on every send."""
    a = _adapter()
    a.MEDIA_GROUP_MAX_WAIT_SECONDS = 3.0  # leaked slot would make the flush wait this out

    a._note_media_group_arrival("g")
    a._note_media_group_arrival("g")
    await a._queue_media_group_event("g", _photo_event("p1"))  # item A queued
    a._media_group_item_finished("g")  # item A's handler exit
    a._media_group_item_finished("g")  # item B exited WITHOUT queueing (matched finally)

    await asyncio.sleep(0.8)  # 0.2 delay + margin; much less than the 3.0 cap

    assert a.handle_message.await_count == 1, "flush waited on a leaked in-flight slot"
    assert _dispatched_urls(a) == [["p1"]]
    assert a._media_group_pending.get("g", 0) == 0, "in-flight counter must settle at zero"


# -- Handler-level regressions (drive the real _handle_media_message path) ---------------


def _handler_adapter():
    """Adapter rig running the real _handle_media_message pipeline (media groups included)."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    a = object.__new__(TelegramAdapter)
    a._media_group_events = {}
    a._media_group_tasks = {}
    a.MEDIA_GROUP_WAIT_SECONDS = 0.2
    a.MEDIA_GROUP_MAX_WAIT_SECONDS = 1.5
    a.handle_message = AsyncMock()
    a._bot_username_observed = "test_bot"
    a._is_user_authorized_from_message = lambda msg: True
    a._should_process_message = lambda msg, **kw: True
    a._media_message_type = lambda msg: MessageType.PHOTO
    a._build_message_event = lambda msg, mt, update_id=None: MessageEvent(
        text=msg.caption or "", message_type=mt)
    a._apply_telegram_group_observe_attribution = lambda event: event
    a._drop_unresolved = lambda event: False
    return a


def _photo_msg(mid, *, delay=0.0, media_group_id="g", caption=None):
    async def _download():
        if delay:
            await asyncio.sleep(delay)
        return bytearray(b"img-bytes")

    file_obj = SimpleNamespace(file_path=f"photos/p{mid}.jpg", download_as_bytearray=_download)
    photo = SimpleNamespace(get_file=AsyncMock(return_value=file_obj))
    return SimpleNamespace(
        message_id=mid, media_group_id=media_group_id, caption=caption, text=None,
        photo=[photo], voice=None, audio=None, video=None, document=None, sticker=None)


def _update(msg):
    return SimpleNamespace(update_id=msg.message_id, message=msg)


def _patch_cache(monkeypatch):
    async def _fake_cache(data, ext=".jpg"):
        return f"cached{ext}"

    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.cache_image_from_bytes_async", _fake_cache)


@pytest.mark.asyncio
async def test_handler_slow_sibling_single_turn(monkeypatch):
    """Two same-media_group updates; the 2nd photo's download outlives the debounce window.
    Exactly one dispatched event, both images — and nothing is dispatched mid-download."""
    _patch_cache(monkeypatch)
    a = _handler_adapter()

    await a._handle_media_message(_update(_photo_msg(1, caption="caption A")), None)
    task2 = asyncio.create_task(a._handle_media_message(_update(_photo_msg(2, delay=0.45)), None))

    await asyncio.sleep(0.3)  # debounce window (0.2) long expired; item 2 still downloading
    assert a.handle_message.await_count == 0, "partial album dispatched while a sibling was still downloading"

    await task2
    await asyncio.sleep(0.6)

    assert a.handle_message.await_count == 1, f"album split into {a.handle_message.await_count} turns"
    urls = _dispatched_urls(a)
    assert urls == [["cached.jpg", "cached.jpg"]]
    assert a._media_group_pending.get("g", 0) == 0


@pytest.mark.asyncio
async def test_handler_early_return_releases_slot(monkeypatch):
    """An item dropped before queueing (unresolved identity) must not leave the flush waiting."""
    _patch_cache(monkeypatch)
    a = _handler_adapter()
    a._drop_unresolved = lambda event: True  # every photo item exits without queueing

    await a._handle_media_message(_update(_photo_msg(1)), None)

    assert a._media_group_pending.get("g", 0) == 0, "early return leaked an in-flight slot"
    assert a.handle_message.await_count == 0
