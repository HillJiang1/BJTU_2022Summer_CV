import tensorflow as tf

def load_preprosess_image(path,label):
    #读取路径
    image=tf.io.read_file(path)
    #解码
    image=tf.image.decode_jpeg(image,channels=3)#彩色图像为3个channel
    #将图像改变为同样的大小，利用裁剪或者扭曲,这里应用了扭曲
    image=tf.image.resize(image,[360,360])
    #随机裁剪图像
    image=tf.image.random_crop(image,[256,256,3])
    #随机上下翻转图像
    image=tf.image.random_flip_left_right(image)
    #随机上下翻转
    image=tf.image.random_flip_up_down(image)
    #随机改变图像的亮度
    image=tf.image.random_brightness(image,0.5)
    #随机改变对比度
    image=tf.image.random_contrast(image,0,1)
    #改变数据类型
    image=tf.cast(image,tf.float32)
    #将图像进行归一化
    image=image/255

    label=tf.reshape(label,[1])
    return image,label