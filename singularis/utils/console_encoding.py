# -*- coding: utf-8 -*-
"""
Console Encoding Utility for Windows UTF-8 Support

This module provides utilities to fix console encoding issues on Windows,
enabling proper display of unicode characters, emojis, and special symbols.

Usage:
    from singularis.utils.console_encoding import ensure_utf8_console
    
    # At the start of your script
    ensure_utf8_console()
"""

import sys
import io


def ensure_utf8_console():
    """Ensures that the console on Windows is configured to use UTF-8 encoding.

    This function addresses common `UnicodeEncodeError` issues that occur when
    printing unicode characters, emojis, or other special symbols to the Windows
    console, which often defaults to a legacy codepage like `cp1252`.

    On Windows, it wraps `sys.stdout` and `sys.stderr` with a `TextIOWrapper`
    that forces UTF-8 encoding and uses 'replace' as the error handler to prevent
    crashes from unprintable characters. The function is idempotent and has no
    effect on non-Windows platforms.

    Returns:
        bool: True if the console encoding was successfully changed, False otherwise.
    """
    if sys.platform != 'win32':
        return False
    
    try:
        # Check if already wrapped with UTF-8
        if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding.lower() == 'utf-8':
            return False
        
        # Wrap stdout and stderr with UTF-8 encoding
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding='utf-8',
            errors='replace',
            line_buffering=True
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer,
            encoding='utf-8',
            errors='replace',
            line_buffering=True
        )
        return True
    except (AttributeError, io.UnsupportedOperation):
        # stdout/stderr doesn't have .buffer (e.g., in some IDEs)
        # or operation not supported
        return False


def print_utf8(*args, **kwargs):
    """A wrapper for the built-in `print` function that safely handles Unicode.

    This function attempts to print the given arguments. If a `UnicodeEncodeError`
    occurs, it falls back to encoding the message as UTF-8 with replacement
    characters for any symbols that cannot be rendered, preventing crashes.

    Args:
        *args: The objects to be printed, same as the built-in `print`.
        **kwargs: Keyword arguments for the built-in `print`.
    """
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Fallback: encode as UTF-8 and decode with errors='replace'
        try:
            message = ' '.join(str(arg) for arg in args)
            print(message.encode('utf-8', errors='replace').decode('utf-8'), **kwargs)
        except Exception:
            # Last resort: print without unicode
            print('[Unicode encoding error - message suppressed]', **kwargs)


def safe_format_unicode(text: str, fallback: str = '[?]') -> str:
    """Safely formats a string to prevent `UnicodeEncodeError` on console output.

    On Windows, this function attempts to encode the string to the console's
    detected encoding. If it fails, it falls back to a UTF-8 encoding with
    replacement characters. On other platforms, it returns the original string.

    Args:
        text: The input string, which may contain unicode characters.
        fallback: The string to use for characters that cannot be encoded.
                  Note: The current implementation uses the default 'replace'
                  behavior, which typically inserts a '?'.

    Returns:
        A string that is safe to print to the console.
    """
    if sys.platform != 'win32':
        return text
    
    try:
        # Try to encode/decode to verify it's safe
        text.encode(sys.stdout.encoding or 'utf-8')
        return text
    except (UnicodeEncodeError, AttributeError):
        # Replace unprintable characters
        return text.encode('utf-8', errors='replace').decode('utf-8')


# Emoji replacements for environments that don't support emoji
ASCII_EMOJI_MAP = {
    '✓': '[OK]',
    '✅': '[OK]',
    '✗': '[X]',
    '❌': '[X]',
    '⚠️': '[!]',
    '🚀': '[>>]',
    '🔄': '[~]',
    '⏳': '[...]',
    '🎮': '[*]',
    '🔴': '[HIGH]',
    '🟡': '[MED]',
    '🟢': '[LOW]',
    '🧠': '[BRAIN]',
    '→': '->',
    '║': '|',
    '═': '=',
    '╔': '+',
    '╚': '+',
    '╗': '+',
    '╝': '+',
}


def replace_emojis_with_ascii(text: str) -> str:
    """Replaces common emojis and unicode symbols with ASCII-friendly equivalents.

    This function is useful for logging or display in environments that have
    poor or non-existent support for unicode characters.

    Args:
        text: The input string, which may contain emojis or special symbols.

    Returns:
        A new string with all recognized unicode symbols replaced by their
        ASCII counterparts.
    """
    for emoji, ascii_equiv in ASCII_EMOJI_MAP.items():
        text = text.replace(emoji, ascii_equiv)
    return text


# Auto-configure on import if environment variable is set
if __name__ != '__main__':
    import os
    if os.getenv('SINGULARIS_AUTO_UTF8', '').lower() in ('1', 'true', 'yes'):
        ensure_utf8_console()
