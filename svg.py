class SVG:
    def __init__(self, head, tail=""):
        self.head = head
        self.tail = tail
        self.contents = []
    
    @staticmethod
    def svg(width = 100, height = 100):
        return SVG(
            '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" \n'
                'width="%fcm" height="%fcm">\n' % (width, height),
            '</svg>')
    
    @staticmethod
    def g():
        return SVG('<g>', '</g>')
         
    @staticmethod
    def gtrans(tx = 0, ty = 0):
        cm = 35.43307
        return SVG(
            '<g transform="translate(%f, %f)">\n' % (tx * cm, ty * cm),
            '</g>')
    
    @staticmethod
    def rect(x, y, width, height, fill="none", stroke="black", stroke_width=0.1, transform=""):
        return SVG('<rect x="%fcm" y="%fcm" width="%fcm" height="%fcm" '
            'fill="%s" stroke="%s" stroke-width="%fcm" transform="%s" stroke-linejoin="bevel"/>\n' %
            (x, y, width, height, fill, stroke, stroke_width, transform))

    @staticmethod
    def text(x, y, text):
        return SVG('<text x="%fcm" y="%fcm">%s</text>\n' % (x, y, text))

    def append(self, content):
        self.contents.append(content)

    def extend(self, conts):
        self.contents.extend(conts)

    def concat(self):
        return (
            self.head +
            "".join(
                content.concat() if type(content) is SVG else content
                for content in self.contents) +
            self.tail
        )

    @staticmethod
    def board(board, size = 1):
        res = SVG.g()
        for y, by in enumerate(board.board[0]):
            for x, bx in enumerate(by):
                if bx != 0:
                    res.append(SVG.rect(y * size, x * size, size, size))
        return res
    
    @staticmethod
    def block(block, size = 0.5):
        res = SVG.g()
        res.append(SVG.text(0, -0.3, block.name))
        skewY = "skewY(45)"
        skewX = "skewX(45)"
        for z, bz in enumerate(block.block):
            for y, by in enumerate(bz):
                for x, bx in enumerate(by):
                    if bx != 0:
                        g = SVG.gtrans((y - 0.5 * z) * size, (x - 0.5 * z) * size)
                        g.append(SVG.rect(size * 2, -size, size * 0.5, size, fill="white", transform=skewY))
                        g.append(SVG.rect(-size, size * 2, size, size * 0.5, fill="white", transform=skewX))
                        g.append(SVG.rect(size, size, size, size, fill="white"))
                        res.append(g)
        return res
    
    @staticmethod
    def blocklist(blocks, size = 0.5):
        res = SVG.g()
        for i, block in enumerate(blocks.blocks):
            g = SVG.gtrans(i * size * 5, 0)
            g.append(SVG.block(block))
            res.append(g)
        return res

    @staticmethod
    def 

    def save(self, f):
        with open(f, "w") as f:
            f.write(self.concat())