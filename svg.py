from nputils import NpUtils
from random import choice

class SVG:
    def __init__(self, head, tail=""):
        self.head = head
        self.tail = tail
        self.contents = []
    
    @staticmethod
    def svg(width = 21, height = 12):
        return SVG(
            '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" \n'
                'width="%fcm" height="%fcm">\n' % (width, height),
            '</svg>')

    @staticmethod
    def html():
        return SVG(
            '<!DOCTYPE html>\n'
                '<title>ubongo</title>'
                '<meta charset="utf-8"><div class="cont">',
        """
        <style>
        .cont{ 
        }
        .probset {
            width: 50cm;
            display: flex;
            flex-wrap: wrap;
            page-break-before: always;
            page-break-after: always;
        }
        .prob {
            border: 1px solid black; 
        }
        </style>
        """
        )
    
    @staticmethod
    def div(cls=""):
        return SVG('<div class="%s">' % cls, '</div>')

    
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
    def rect(x, y, width, height, fill="none", stroke="black", stroke_width=0.05, transform=""):
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

    name_colors = {"G": "#00ff09", "Y": "#ffd500", "R": "#ff0000", "B": "#1e00ff"}
    blocksize = 1.4 * (7.0 / 4.7)
    @staticmethod
    def board(board):
        size = SVG.blocksize
        res = SVG.g()
        bz = NpUtils.shirnk(board.board)[0]
        colors = list(SVG.name_colors.values())
        for y, by in enumerate(reversed(bz)):
            for x, bx in enumerate(by):
                if bx != 0:
                    res.append(SVG.rect(y * size, x * size, size, size, fill="#ddd", stroke=choice(colors)))
        return res
    
    minisize = 0.2
    @staticmethod
    def block(block):
        size = SVG.minisize
        res = SVG.g()
        #res.append(SVG.text(0, -0.3, block.name))
        skewY = "skewY(45)"
        skewX = "skewX(45)"
        color = SVG.name_colors[block.name[0]]
        stroke_width = 0.03
        for z, bz in enumerate(block.block):
            for y, by in enumerate((bz)):
                for x, bx in enumerate(reversed(by)):
                    if bx != 0:
                        g = SVG.gtrans((y - 0.5 * z) * size, (x - 0.5 * z) * size)
                        g.append(SVG.rect(size * 2, -size, size * 0.5, size,
                                          fill=color, stroke_width=stroke_width, transform=skewY))
                        g.append(SVG.rect(-size, size * 2, size, size * 0.5,
                                          fill=color, stroke_width=stroke_width, transform=skewX))
                        g.append(SVG.rect(size, size, size, size,
                                          fill=color, stroke_width=stroke_width))
                        res.append(g)
        return res
    
    @staticmethod
    def blocklist(blocks):
        size = SVG.minisize
        res = SVG.g()
        for i, block in enumerate(blocks.blocks):
            g = SVG.gtrans(i * size * 5, 0)
            g.append(SVG.block(block))
            res.append(g)
        return res

    @staticmethod
    def problemboard(probs):
        size = SVG.blocksize
        minisize = SVG.minisize
        svg = SVG.svg()
        res = SVG.gtrans(2,2)
        res.append(SVG.board(probs.board))
        res.append(SVG.text(0, -0.5, probs.name))
        bl = SVG.gtrans(size * 6.5, 0)
        for i, prob in enumerate(probs.problems):
            g = SVG.gtrans(0, i * minisize * 8)
            g.append(SVG.text(-0.5, 0.5, str(i+1)))
            g.append(SVG.blocklist(prob.blocks))
            bl.append(g)
        res.append(bl)
        svg.append(res)
        return svg
    
    @staticmethod
    def problemset(probset):
        res = SVG.div("probset")
        for _, probb in enumerate(probset.probboards):
            div = SVG.div("prob")
            svg = SVG.problemboard(probb)
            div.append(svg)
            res.append(div)
        return res

    @staticmethod
    def game(game):
        html = SVG.html()
        for probs in game.probsets:
            html.append(SVG.problemset(probs))
        return html

    @staticmethod
    def game_tate(game):
        html = SVG.html()
        pss = [[] for _ in range(len(game.probsets[0].probboards))]
        for probs in game.probsets:
            for i, probb in enumerate(probs.probboards):
                div = SVG.div("prob")
                svg = SVG.problemboard(probb)
                div.append(svg)
                pss[i].append(svg)
        for ps in pss:
            div = SVG.div("probset")
            for i, p in enumerate(ps):
                if i != 0 and i % 4 == 0:
                    html.append(div)
                    div = SVG.div("probset")
                div.append(p)
            html.append(div)
        return html

    def save(self, f):
        with open(f, "w") as f:
            f.write(self.concat())