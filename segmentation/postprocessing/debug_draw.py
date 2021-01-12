import itertools

from PIL import ImageDraw

from segmentation.preprocessing.source_image import SourceImage


class DebugDrawDummy:
    def __init__(self, *args, **kwargs):
        pass

    def draw_bboxs(self, bboxs):
        pass

    def draw_baselines(self, baselines):
        pass

    def draw_polygons(self, polys):
        pass

    def image(self):
        raise NotImplementedError("requesting image but drawing is disables")


class DebugDraw:
    colors = [(255, 0, 0),
              (0, 255, 0),
              (0, 0, 255),
              (255, 255, 0),
              (0, 255, 255),
              (255, 0, 255)]

    def __init__(self, source_image: SourceImage):
        self.img = source_image.pil_image.copy().convert('RGB')
        self.draw = ImageDraw.Draw(self.img)

    def draw_bboxs(self, bboxs):
        for ind, x in enumerate(bboxs):
            if x.bbox:
                self.draw.line(x.bbox + [x.bbox[0]], fill=DebugDraw.colors[ind % len(DebugDraw.colors)], width=3)
                self.draw.text((x.bbox[0]), "type:{}".format(x.baselines[0].cluster_type))

    def draw_baselines(self, baselines):
        if baselines is None or len(baselines) == 0:
            return
        for ind, x in enumerate(baselines):
            t = list(itertools.chain.from_iterable(x))
            a = t[::]
            self.draw.line(a, fill=DebugDraw.colors[ind % len(DebugDraw.colors)], width=4)

    def draw_polygons(self, polys):
        for ind, x in enumerate(polys):
            l = list(itertools.chain.from_iterable(x))
            self.draw.polygon(l, outline=DebugDraw.colors[ind % len(DebugDraw.colors)])

    def image(self):
        return self.img
