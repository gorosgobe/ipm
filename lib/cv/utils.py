class CvUtils(object):
    @staticmethod
    def fit_crop_to_image(center_x, center_y, height, width, cropped_height, cropped_width):
        half_size_height = cropped_height // 2
        half_size_width = cropped_width // 2
        dx = 0
        dy = 0
        if center_x + half_size_width >= width:
            dx = -(center_x + half_size_width - width) - 1
        elif center_x - half_size_width < 0:
            dx = -(center_x - half_size_width)
        if center_y + half_size_height >= height:
            dy = -(center_y + half_size_height - height) - 1
        elif center_y - half_size_height < 0:
            dy = -(center_y - half_size_height)
        # otherwise, crop lies fully inside the image, dx, dy = 0 apply
        center_x += dx
        center_y += dy
        return center_x, center_y

    @staticmethod
    def get_bounding_box_coordinates(center_x, center_y, cropped_height, cropped_width):
        half_size_height = cropped_height // 2
        half_size_width = cropped_width // 2
        y_min = center_y - half_size_height + (cropped_height % 2 == 0)
        y_max = center_y + half_size_height
        x_min = center_x - half_size_width + (cropped_width % 2 == 0)
        x_max = center_x + half_size_width
        return x_min, x_max, y_min, y_max

    @staticmethod
    def get_bounding_box(x_min, x_max, y_min, y_max):
        return [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]
