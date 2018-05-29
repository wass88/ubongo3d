class SVG:
    def full_svg(svg, width = 100, height = 100):
    return ('<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="%dpx" height="%dpx">' %
        (width, height) +
        svg +
    '</svg>')

    def svg_board(board, size = 10, width = 2):
    res = []
    for by, y in enumerate(board):
        for bx, x in enumerate(by):
        if bx != 0:
            res += ('<rect x="%d" y="%d" width="%d" height="%d"'
            'fill="none" stroke="black" stroke-width="%d" />' %
            (y * size, x * size, size, size, 2)
            )
    return res