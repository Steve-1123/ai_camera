from __future__ import annotations

from textwrap import wrap
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from app.visualization.pose_renderer import PoseRenderer

CARD_SIZE = (900, 1200)


class PoseCardRenderer:
    def __init__(self, pose_renderer: PoseRenderer | None = None) -> None:
        self.pose_renderer = pose_renderer or PoseRenderer()
        self.title_font = _load_font(46)
        self.section_font = _load_font(25)
        self.body_font = _load_font(28)
        self.tag_font = _load_font(22)

    def render_pose_card(self, pose: dict[str, Any]) -> Image.Image:
        card = Image.new("RGB", CARD_SIZE, (248, 250, 252))
        draw = ImageDraw.Draw(card)

        draw.rounded_rectangle([30, 30, 870, 1170], radius=24, fill="white", outline=(226, 232, 240), width=2)
        draw.text((72, 70), str(pose.get("display_name", "Pose")), fill=(15, 23, 42), font=self.title_font)

        skeleton = self.pose_renderer.render_skeleton(pose.get("landmarks"))
        skeleton.thumbnail((650, 650))
        skeleton_x = (CARD_SIZE[0] - skeleton.width) // 2
        card.paste(skeleton, (skeleton_x, 150))

        y = 840
        draw.text((72, y), "Instructions", fill=(71, 85, 105), font=self.section_font)
        y += 42
        instructions = pose.get("instructions") if isinstance(pose.get("instructions"), list) else []
        for index, instruction in enumerate(instructions[:3], start=1):
            text = str(instruction)
            for line_index, line in enumerate(_wrap_text(text, 32)):
                prefix = f"{index}. " if line_index == 0 else "   "
                draw.text((92, y), prefix + line, fill=(15, 23, 42), font=self.body_font)
                y += 36
            y += 8

        tags = _string_list(pose.get("scene_tags")) + _string_list(pose.get("style_tags"))
        if tags:
            y += 18
            draw.text((72, y), "Tags", fill=(71, 85, 105), font=self.section_font)
            self._draw_tags(draw, tags[:8], y + 48)

        return card

    def _draw_tags(self, draw: ImageDraw.ImageDraw, tags: list[str], start_y: int) -> None:
        x = 72
        y = start_y
        for tag in tags:
            label = f"#{tag}"
            bbox = draw.textbbox((0, 0), label, font=self.tag_font)
            width = bbox[2] - bbox[0] + 32
            if x + width > 828:
                x = 72
                y += 46
            draw.rounded_rectangle([x, y, x + width, y + 34], radius=17, fill=(239, 246, 255), outline=(191, 219, 254))
            draw.text((x + 16, y + 4), label, fill=(29, 78, 216), font=self.tag_font)
            x += width + 12


def _load_font(size: int) -> ImageFont.ImageFont:
    for font_path in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_text(text: str, width: int) -> list[str]:
    lines = wrap(text, width=width)
    return lines or [text]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]
