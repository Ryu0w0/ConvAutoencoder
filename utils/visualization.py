import numpy as np
from torchvision import utils as vutils
from utils import file_operator as f_op


def save_image_as_grid(img_tensor, save_key, file_name):
    img_tensor = img_tensor.detach()
    save_image_path = f"./files/output/images/{save_key}"
    f_op.create_folder(save_image_path)
    if img_tensor.shape[0] > 8:
        img_tensor = img_tensor[0:8, :, :, :]
    img_array = vutils.make_grid(img_tensor, padding=2, normalize=False).cpu().numpy()
    img_array = np.transpose(img_array, (1, 2, 0))
    f_op.save_as_jpg(img_array * 255, save_root_path=save_image_path,
                     save_key=save_key, file_name=file_name)
# TODO: 保存タイミング指定したい