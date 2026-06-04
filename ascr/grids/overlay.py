from pathlib import Path


def _write_ppm_grid(output_path, image_size=256, grid_size=4):
    output_path = Path(output_path)
    lines = ["P3", f"{image_size} {image_size}", "255"]
    step = max(1, image_size // grid_size)
    for row in range(image_size):
        pixels = []
        for col in range(image_size):
            if row % step == 0 or col % step == 0 or row == image_size - 1 or col == image_size - 1:
                pixels.append("255 255 255")
            else:
                base = 32 + ((row // step) * 35 + (col // step) * 25) % 160
                pixels.append(f"{base} {max(0, base - 20)} {min(255, base + 50)}")
        lines.append(" ".join(pixels))
    output_path.write_text(chr(10).join(lines) + chr(10), encoding="ascii")
    return output_path


def create_grid_overlay(input_path, output_path, image_size=256, grid_size=4):
    output_path = Path(output_path)
    try:
        from PIL import Image, ImageDraw
        image = Image.open(input_path).convert("RGB").resize((image_size, image_size))
        draw = ImageDraw.Draw(image)
        step = image_size // grid_size
        for index in range(grid_size + 1):
            pos = min(image_size - 1, index * step)
            draw.line((pos, 0, pos, image_size), fill=(255, 255, 255), width=2)
            draw.line((0, pos, image_size, pos), fill=(255, 255, 255), width=2)
        for row in range(grid_size):
            for col in range(grid_size):
                label_char = chr(ord("A") + row)
                label = f"{label_char}{col + 1}"
                draw.text((col * step + 5, row * step + 5), label, fill=(255, 255, 255))
        image.save(output_path)
        return output_path
    except Exception:
        return _write_ppm_grid(output_path, image_size=image_size, grid_size=grid_size)


def _write_ppm_token_grid(output_path, image_size=512, token_grid_size=32, label_step=4):
    output_path = Path(output_path)
    lines = ["P3", f"{image_size} {image_size}", "255"]
    step = max(1, image_size // token_grid_size)
    major = max(1, step * label_step)
    for row in range(image_size):
        pixels = []
        for col in range(image_size):
            on_major = row % major == 0 or col % major == 0
            on_minor = row % step == 0 or col % step == 0
            on_edge = row == image_size - 1 or col == image_size - 1
            if on_major or on_edge:
                pixels.append("255 255 255")
            elif on_minor:
                pixels.append("120 120 120")
            else:
                base = 32 + ((row // step) * 7 + (col // step) * 11) % 160
                pixels.append(f"{base} {max(0, base - 20)} {min(255, base + 50)}")
        lines.append(" ".join(pixels))
    output_path.write_text(chr(10).join(lines) + chr(10), encoding="ascii")
    return output_path


def create_token_grid_overlay(input_path, output_path, image_size=512, token_grid_size=32, label_step=4):
    """Overlay a token-resolution reference grid (default 32x32).

    Minor lines mark every token boundary; thicker labelled major lines (and
    numeric row/col ticks) are drawn every ``label_step`` tokens so the semantic
    evaluator can localize a token coordinate ``R{row}C{col}`` (0-indexed) when
    selecting which discrete image tokens are wrong. Existing
    :func:`create_grid_overlay` is intentionally left unchanged.
    """
    output_path = Path(output_path)
    try:
        from PIL import Image, ImageDraw
        image = Image.open(input_path).convert("RGB").resize((image_size, image_size))
        draw = ImageDraw.Draw(image)
        step = image_size / float(token_grid_size)
        for index in range(token_grid_size + 1):
            pos = min(image_size - 1, int(round(index * step)))
            is_major = index % label_step == 0 or index == token_grid_size
            color = (255, 255, 255) if is_major else (120, 120, 120)
            width = 2 if is_major else 1
            draw.line((pos, 0, pos, image_size), fill=color, width=width)
            draw.line((0, pos, image_size, pos), fill=color, width=width)
        for index in range(0, token_grid_size, label_step):
            pos = int(round(index * step)) + 2
            draw.text((pos, 1), str(index), fill=(255, 255, 0))
            draw.text((1, pos), str(index), fill=(255, 255, 0))
        image.save(output_path)
        return output_path
    except Exception:
        return _write_ppm_token_grid(output_path, image_size=image_size, token_grid_size=token_grid_size, label_step=label_step)
