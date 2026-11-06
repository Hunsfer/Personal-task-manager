"""aiogram wrapper, and the only module allowed to import it.

This uses the bot's own token. It has no access to the owner's Telegram session.
"""

import asyncio
import logging
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.exceptions import (
    TelegramAPIError,
    TelegramBadRequest,
    TelegramNetworkError,
    TelegramRetryAfter,
)
from aiogram.types import (
    BotCommand,
    BotCommandScopeChat,
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from aiogram.types import Message as TelegramMessage

from task_manager.review_bot.controller import Controller
from task_manager.review_bot.messenger import (
    COMMANDS,
    Keyboard,
    MessageGone,
    MessengerError,
    RetryLater,
)

logger = logging.getLogger(__name__)


def _markup(keyboard: Keyboard | None) -> InlineKeyboardMarkup | None:
    if not keyboard:
        return None
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=b.text, callback_data=b.data) for b in row]
            for row in keyboard
        ]
    )


class TelegramMessenger:
    def __init__(self, token: str, owner_id: int) -> None:
        self._owner_id = owner_id
        self._bot = Bot(
            token,
            default=DefaultBotProperties(parse_mode="HTML", link_preview_is_disabled=True),
        )

    async def close(self) -> None:
        await self._bot.session.close()

    async def _publish_commands(self) -> None:
        """Only the owner's chat gets the menu. Without it the bot still works, so never fatal."""
        try:
            await self._bot.set_my_commands(
                [BotCommand(command=name, description=text) for name, text in COMMANDS],
                scope=BotCommandScopeChat(chat_id=self._owner_id),
            )
        except (TelegramAPIError, TelegramNetworkError):
            logger.warning("bot_commands_not_set")

    async def delete(self, message_id: int) -> None:
        """Telegram refuses this for messages older than 48 hours, which raises MessageGone."""
        try:
            await self._bot.delete_message(self._owner_id, message_id)
        except TelegramRetryAfter as error:
            raise RetryLater(error.retry_after) from None
        except TelegramBadRequest:
            raise MessageGone("message can't be deleted") from None
        except (TelegramAPIError, TelegramNetworkError) as error:
            raise MessengerError(type(error).__name__) from None

    async def send(self, text: str, keyboard: Keyboard | None = None, silent: bool = False) -> int:
        try:
            message = await self._bot.send_message(
                self._owner_id,
                text,
                reply_markup=_markup(keyboard),
                disable_notification=silent,
            )
        except TelegramRetryAfter as error:
            raise RetryLater(error.retry_after) from None
        except (TelegramAPIError, TelegramNetworkError) as error:
            raise MessengerError(type(error).__name__) from None
        return message.message_id

    async def send_file(self, path: Path, name: str) -> int:
        try:
            message = await self._bot.send_document(
                self._owner_id, FSInputFile(path, filename=name)
            )
        except TelegramRetryAfter as error:
            raise RetryLater(error.retry_after) from None
        except (TelegramAPIError, TelegramNetworkError) as error:
            raise MessengerError(type(error).__name__) from None
        return message.message_id

    async def edit(self, message_id: int, text: str, keyboard: Keyboard | None = None) -> None:
        try:
            await self._bot.edit_message_text(
                text, chat_id=self._owner_id, message_id=message_id, reply_markup=_markup(keyboard)
            )
        except TelegramRetryAfter as error:
            raise RetryLater(error.retry_after) from None
        except TelegramBadRequest as error:
            description = str(error.message).lower()
            if "not modified" in description:
                return
            if "not found" in description or "can't be edited" in description:
                raise MessageGone("message is gone") from None
            raise MessengerError(type(error).__name__) from None
        except (TelegramAPIError, TelegramNetworkError) as error:
            raise MessengerError(type(error).__name__) from None

    async def run(self, controller: Controller, stop: asyncio.Event) -> None:
        """Long polling, so nothing listens on a port."""
        await self._publish_commands()
        dispatcher = Dispatcher()

        @dispatcher.message(F.text)
        async def on_message(message: TelegramMessage) -> None:
            user = message.from_user.id if message.from_user else None
            await controller.on_text(user, message.text or "", message.message_id)

        @dispatcher.callback_query()
        async def on_callback(query: CallbackQuery) -> None:
            # Only the owner sees that the press was received.
            if controller.is_owner(query.from_user.id):
                await query.answer()
            await controller.on_callback(query.from_user.id, query.data or "")

        polling = asyncio.create_task(dispatcher.start_polling(self._bot, handle_signals=False))
        waiting = asyncio.create_task(stop.wait())
        done, _ = await asyncio.wait({polling, waiting}, return_when=asyncio.FIRST_COMPLETED)
        if polling in done:
            waiting.cancel()
            await polling
            return
        await dispatcher.stop_polling()
        await polling
