




class imdb(object):
	def __init__(self):
		self._classes = []
        self._image_index = []

		pass
    @property
    def num_images(self):
      return len(self.image_index)

    @property
    def image_index(self):
        return self._image_index

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def roidb(self):
    	return self.roidb_handler()

	def roidb_handler(self):
		pass

