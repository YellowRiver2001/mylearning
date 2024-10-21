关于AI-ISP的调研：  
一，ISP（图像信号处理）  
ISP（Image Signal Process，图像信号处理），即下图中成像引擎。  
![image](https://github.com/user-attachments/assets/692eb4a3-5868-4678-b2bb-3f1851fd65a6)  

    1.主流的CMOS和CCD sensor几乎都是输出Bayer mosaic格式的RAW数据，这种数据格式是无法直接观看的，  
      必须转换成常见的RGB或YUV格式才能被主流的图像处理软件支持。  
    2.对于camera产品而言，一般还需要将RGB或YUV图像进一步转换成JPEG格式以方便进行存储。  
      上述图像处理过程统称图像信号处理（Image Signal Processing，ISP）。  
    3.广义的ISP包含了JPEG和H.264/265图像压缩处理，而狭义的ISP仅包括从RAW格式变换到RGB或YUV的处理过程。  
    
 由于镜头和Sensor的物理缺陷(不完美)，需要ISP模块去补偿。
 ISP主要做下面的工作：  
 
     AEC（自动曝光控制）、AGC（自动增益控制）、AWB（自动白平衡）、  
     颜色校正、Lens Shading、Gamma 校正、祛除坏点、Auto Black Level、Auto White Level  

传统ISP流程图：  
![image](https://github.com/user-attachments/assets/70d81c98-16a1-4620-b5dc-3ef4e0302d90)


AEC( Automatic Exposure Control )  

        自动曝光控制是指根据光线的强弱自动调整曝光量，防止曝光过度或者不足，在不同的照明条件和场景中实现欣  
        赏亮度级别或所谓的目标亮度级别，从而捕获的视频或图像既不太暗也不太亮  
![image](https://github.com/user-attachments/assets/9e8d0f5e-1b11-4086-8a86-754e99cd45cb)  

HDR ( High-Dynamic Range Imaging ) 高动态范围成像  

         Sensor的动态范围就是Sensor在一幅图像里能够同时体现高光和阴影部分内容的能力。在自然界的真实情况，  
     有些场景的动态范围要大于100 dB，人眼的动态范围可以达到100dB。高动态范围成像的目的就是要正确地表   
     示真实世界中的亮度范围。适合场景：比较适合在具有背光的高对比度场景下使用如：日落、室内窗户，这样   
     能使明处的景物不致过曝，而使得暗处的景物不致欠曝。  
![image](https://github.com/user-attachments/assets/75f84ce9-04ab-4948-bad5-3283ca28e1cc)  

AWB ( Auto White Balance ) 自动白平衡  

            白平衡，顾名思义，即白色的平衡，由于人眼的适应性，在不同色温下，都能准确判断出白色，  
        但是相机就差远了，在不同色温的光源下，图像会出现偏色，与人眼看到的颜色不一致，因此需要进行白平衡处理。  
![image](https://github.com/user-attachments/assets/3b2445e9-03ea-41ba-92c7-72bf01967b85)   
![image](https://github.com/user-attachments/assets/5b48317d-9f0d-44f6-b06f-77dcf31801cb)  
![image](https://github.com/user-attachments/assets/ddb32c08-8fee-4f7f-9eba-a952464b43cf)


 CCM ( Color Correction Matrix ) 颜色校正  
 
         颜色校正主要为了校正在滤光板处各颜色块之间的颜色渗透带来的颜色误差。一般颜色校正的过程是  
     首先利用该图像传感器拍摄到的图像与标准图像相比较，以此来计算得到一个校正矩阵。该矩阵就是  
     该图像传感器的颜色校正矩阵。在该图像传感器应用的过程中，及可以利用该矩阵对该图像传感器所  
     拍摄的所有图像来进行校正，以获得最接近于物体真实颜色的图像    
     ![image](https://github.com/user-attachments/assets/9a51054d-8411-4ff4-b774-103db15d7144)

![image](https://github.com/user-attachments/assets/f122df8b-4bd8-48a0-a7e3-c70eab3f0458)  

DNS ( Denoise ) 去噪  

        使用 CMOS Sensor 获取图像，光照程度和传感器问题是生成图像中大量噪声的主要因素。同时，
        当信号经过 ADC 时，又会引入其他一些噪声。这些噪声会使图像整体变得模糊，而且丢失很多细节，所以需要对图像进行去噪处理空间去噪传统的方法有均值滤波、高斯滤波  
![image](https://github.com/user-attachments/assets/0b6b7d87-c6a2-4353-b5a9-07658038d861)  

BLC ( Black Level Correction ) 黑电平校正   

        Black Level 是用来定义图像数据为 0 时对应的信号电平。由于暗电流的影响，传感器出来的实际原始数据并不是我们需要的黑平衡。为减少暗电流对图像信号的影响，采用的方法是从已获得的图像信号中减去参考暗电流信号。一般情况下,在传感器中，实际像素要比有效像素多,像素区头几行作为不感光区，用于自动黑电平校正,其平均值作为校正值，然后在下面区域的像素都减去此矫正值，那么就可以将黑电平矫正过来了。
![image](https://github.com/user-attachments/assets/7ec458b5-75f5-4258-b62e-1c27bab49af3)

 LSC ( Lens Shade Correction ) 镜头阴影校正  

     由于相机在成像距离较远时，随着视场角慢慢增大，能够通过照相机镜头的斜光束将慢慢减少，从而使得获得的图像中间比较亮，边缘比较暗，这个现象就是光学系统中的渐晕。由于渐晕现象带来的图像亮度不均会影响后续处理的准确性。因此从图像传感器输出的数字信号必须先经过镜头矫正功能块来消除渐晕给图像带来的影响。
![image](https://github.com/user-attachments/assets/e7b0fce4-bed2-424a-b31d-97474496f373)

伽马矫正：  

        相机传感器读数与接收到的光成线性比例，但人类视觉系统自然不会线性感知光，而是对较暗的区域更敏感。  
    因此，通常的做法是用伽马对数函数来调整线性传感器读数  


二， RAW  
1.概念  

        记录了 CMOS 或者 CCD 图像传感器将捕捉到的光源（光子）信号转化为数字信号的原始数据，  
        同时记录了由相机拍摄所产生的一些原数据（Metadata，如 ISO 的设置、快门速度、光圈值、白平衡等）  
        的文件是未经处理、也未经压缩的格式，可以把 RAW 看做数字底片。如下图：    
![image](https://github.com/user-attachments/assets/e1a0b501-4ba3-452f-9048-a3ee5c7065b4)

2 RAW 数据  
详细内容参考：https://www.ruanyifeng.com/blog/2012/12/bayer_filter.html   

        2.1位深：按照每个像素点亮度记录精度（位深）的不同，区分为 RAW10，RAW12、RAW14等，越大的位深意味着更强的颜色表征能力，  
                 比如一个 16bit 的模组拥有 65536 种不同的亮度级别  
        2.2内存占用：但都是每两个字节（16bits）存储一个像素的亮度值，有效 bit 位数分别为 10、12、14，无效 bit 位用 0 补齐，  
                     所以这三种 RAW 数据的大小都是 宽x高x2 个字节  
        2.3模式：传感器只能感应到光照强度的大小，这意味只能是获取黑白 (0,1) 照片，但是现在大部分照片都是彩色的，这是怎么回事呢？  
                 原来有一个叫 Bayer 的人发明了一种彩色滤光片阵列（最常用的是 Bayer 阵列），其巧妙地将这个矩阵加持在传感器上，  
                 只让相应颜色波长的光子通过。仿照了人眼对于颜色的特殊模式要求，到此即形成了不同模式的 Raw 图（一般 BAYER 格式分为 GBRG、GRBG、BGGR、RGGB 四种模式）  
![image](https://github.com/user-attachments/assets/1702301e-7783-44f1-8921-1ba0aa97ad67)  
![image](https://github.com/user-attachments/assets/54787b0d-c9ab-4a24-8006-5c15140acf64)  






二，AI-ISP: 


![image](https://github.com/user-attachments/assets/08ea6b83-ef21-4d9b-9885-b6f8f109fe24)  
红色部分是具有挑战性的部分，AI可以在这些部分取得显著进展：  AWB（光照估计），去马赛克，降噪，超分辨率

1.超分辨率（Super-resolution)  
![image](https://github.com/user-attachments/assets/b717d794-b3c9-4c18-b5df-9f74445c1c89)  
（1）Accurate Image Super-Resolution Using Very Deep Convolutional Networks  
![image](https://github.com/user-attachments/assets/b3495c52-3fc1-47b6-967e-4f0f6933973d)  
•成对的卷积层+非线性激活
•将预测添加到上采样的低分辨率输入中
•梯度裁剪


2.自动白平衡（AWB）：  
（1）FC4:Fully Convolution Color Constancy with Confidence-weighted Pooling(CVPR'17）  
![image](https://github.com/user-attachments/assets/e64857f9-e44a-4baa-9d89-f51f73f0e6a9)  

 
3.去马赛克（Demosiacing）  
（1）Joint Demosaicing and Denoising with Self Guidance  
![image](https://github.com/user-attachments/assets/5ee90a8a-4fce-4115-9b21-f8f47026f282)  


4.降噪  
（1）Beyond a Gaussian Denoiser Residual Learning of Deep CNN for Image Denoising. (TIP'17)  
![image](https://github.com/user-attachments/assets/bef4f30d-f375-4fa8-bc3f-e68855fe5824)  
基于深度残差学习的直接网络（Kim SR ResNet）将批归一化引入网络。-预测残余噪声层








 
1.直接用End-to-End的端到端网络代替传统ISP  
![image](https://github.com/user-attachments/assets/2d678c1d-1c57-428a-b3a5-207f94c16fff)  



1.1 One stage DNN 网络  
（1）Modlling the Scene Dependent Imaging in Cameras with a Deep Nerual Network (CVPR'17)   

        本文的目的是将ISP从sRGB“逆转”为RAW，解决了ISP在辐射校准中的场景依赖性  
        但是，该框架可用于“正向渲染”（RAW到sRGB） 

![image](https://github.com/user-attachments/assets/9dd76913-1d54-431d-b222-152f6e546e3e)

        本文研究图像补丁，并针对每个相机进行训练。为了对局部上下文信息进行编码，使用了一个可学习的直方图特征，  
        并在不同尺度上进行轮询。局部直方图功能为从RAW转换为sRGB（或从sRGB转换为RAW）提供了空间上下文  
（2)Learning to see in the dark (CVPR'18)   

        利用深度神经网络进行数据驱动的图像处理，改进极端低光环境中的快速成像质量。通过端到端训练，该方法整合颜色转换、去马赛克、降噪及图像增强等功能，直接从低光原始数据重建高清图像，有效控制噪声，优化了传统技术的局限。  
![image](https://github.com/user-attachments/assets/d99a36ff-8061-492d-be26-4d9f4935f872)  

    对于拜耳阵列，我们将输入打包成四个通道，并在每个维度上将空间分辨率减半,减去黑色水平并按所需放大比例（例如x100或x300）对数据进行缩放。打包和放大的数据被输入到一个全卷积网络（默认U-Net)中。输出是一个12通道的图像，空间分辨率减半。这个半尺寸的输出通过亚像素层处理以恢复原始分辨率。  

U-Net：  
![image](https://github.com/user-attachments/assets/4e8574c2-490a-4701-8199-96b233a05857)  

黑色水平：  

        “黑色水平”通常指的是相机传感器在捕捉图像时的黑电平校正（black level correction）。相机传感器在没有光照（即完全黑暗）下仍然会有一些电荷积累，这会导致捕捉的图像数据中存在一些非零的偏移值，这就是黑电平。这些偏移值需要在后续图像处理过程中校正，以确保图像数据的准确性。  
        

效果展示：  
![image](https://github.com/user-attachments/assets/a72a85d2-b769-4e59-904b-6ff524ee900c)

1.2  A two stage DNN-based ISP  
（1）CameraNet：A Two-Stage Framework for Effective Camera ISP Learning (TIP2021)  
网络架构：  
![image](https://github.com/user-attachments/assets/98dfc6a5-bdbc-45d9-b18b-ff6f6ee874ad)
实验结果：  
![image](https://github.com/user-attachments/assets/3eb109f7-8371-4d6b-b243-8f3a00896a81)

评价指标：  
1.PSNR（峰值信噪比）：衡量图像或视频质量的常用指标。
PSNR是基于MSE(均方误差)定义，对给定一个大小为m*n的原始图像I和对其添加噪声后的噪声图像K，其MSE可定义为：  
![image](https://github.com/user-attachments/assets/c6a4fbbf-e9f1-4e04-8b7d-02c5d12a9f6f)  
则PSNR可定义为：  
![image](https://github.com/user-attachments/assets/d4a691dd-6aa1-401b-bb49-bfff32526b03)  

        其中MAXI为图像的最大像素值，PSNR的单位为dB。若每个像素由8位二进制表示，则其值为2^8-1=255。但注意这  
        是针对灰度图像的计算方法，若是彩色图像，通常可以由以下方法进行计算：计算RGB图像三个通道每个通道的MSE值  
        再求平均值，进而求PSNR。
        PSNR值越大，表示图像的质量越好，一般来说：
        意义：
        PSNR接近 50dB ，代表压缩后的图像仅有些许非常小的误差。  
        PSNR大于 30dB ，人眼很难查觉压缩后和原始影像的差异。  
        PSNR介于 20dB 到 30dB 之间，人眼就可以察觉出图像的差异。  
        PSNR介于 10dB 到 20dB 之间，人眼还是可以用肉眼看出这个图像原始的结构，且直观上会判断两张图像不存在很大的差异。  
        PSNR低于 10dB，人类很难用肉眼去判断两个图像是否为相同，一个图像是否为另一个图像的压缩结果。  


2.SSIM：  

        SSIM全称为Structural Similarity，即结构相似性，用于评估两幅图像相似度的指标，常用于衡量图像失真前与失真  
        后的相似性，也用于衡量模型生成图像的真实性。  
计算方法：  

        SSIM的计算基于滑动窗口实现，即每次计算均从图片上取一个尺寸为N × N 的窗口，一般N取11，基于窗口计算SSIM指标，  
        遍历整张图像后再将所有窗口的数值取平均值，作为整张图像的SSIM指标。  
        假设x xx表示第一张图像窗口中的数据，y yy表示第二张图像窗口中的数据。其中图像的相似性由三部分构成：  
        luminance(亮度)、contrast(对比度)和structure(结构)。  
luminance计算公式为：  
![image](https://github.com/user-attachments/assets/dd6d04e0-f807-4d29-a4d1-8254f270bce9)
contrast计算公式为：  
![image](https://github.com/user-attachments/assets/0ca2a1bc-0148-4010-b12c-b4db2506ace9)  
structure计算公式为：  
![image](https://github.com/user-attachments/assets/e53d67aa-fb6f-48d0-87d6-b74e1365de34)  
![image](https://github.com/user-attachments/assets/5af84131-35fd-4250-86b9-684a7cb7f888)  

3.MS-SSIM：
MS-SSIM（Multi-Scale Structural Similarity Index）是一种用于评估图像质量的指标，它是结构相似性指数（SSIM）在多个尺度上的扩展。

SSIM是一种衡量两幅图像相似性的指标，它考虑了图像的亮度、对比度和结构等方面。而MS-SSIM在SSIM的基础上引入了多个尺度，以更好地捕捉图像的细节信息。

具体而言，MS-SSIM的计算过程如下：

将原始图像和重建图像划分为不同尺度的子图像。

对每个尺度的子图像计算SSIM指数。

对每个尺度的SSIM指数进行加权平均，得到最终的MS-SSIM值。

MS-SSIM的值范围在0到1之间，数值越接近1表示重建图像与原始图像的相似度越高，图像质量越好。

相比于PSNR，MS-SSIM考虑了图像的结构信息，能够更好地反映人眼对图像质量的感知。它在评估图像质量方面具有更高的准确性和敏感性。

需要注意的是，MS-SSIM计算复杂度相对较高，因为它需要对图像进行多尺度的分解和计算。然而，由于其良好的性能，在图像压缩、图像处理等领域得到广泛应用，并且被认为是一种较为可靠的图像质量评估指标。

1.3 Three stage DNN-based ISP  
(1)Deep-FlexISP: A Three-Stage Framework for Night Photography Rendering.(CVPRW'22 Xiaomi)  
网络结构：  
![image](https://github.com/user-attachments/assets/0fda782a-acab-4bc4-a7f1-3697eaca7384)  

        总体结构包括三个网络：去噪网络、白平衡网络和拜耳阵列（RAW）到sRGB网络。输入是相机捕获的原始数据，输出是sRGB图像 。对于去噪网络，我们使用具有残差块的3级U-Net、平均池化层和元素简单相加。对于WB（白平衡）网络，我们使用FC4，并附加一个去马赛克层和CCM映射层。对于bayer2rgb网络，我们使用没有上采样层的MW-ISP网络。
        许多研究表明，原始域的去噪性能优于RGB域。在原始域中去噪可以更好地去除噪声并保留更多细节。因此，该模型首先将原始原始图像放入去噪网络中，以获得无噪声的原始图像。其次，去噪后的原始图像将通过白平衡（WB）网络来估计RGB增益参数。颜色校正后，拜耳到sRGB（bayer2rgb）网络将原始图像映射到最终输出的sRGB图像。拜耳到sRGB的映射包括去马赛克、色调映射等。   
        分解成三个网络的原因：  
        ISP系统中的一些模块相关性较弱。去噪模块控制噪声强度，白平衡模块控制颜色，色调映射模块控制全局和局部亮度。单级网络很难训练以适应复杂的ISP，特别是夜间照片。任务分解的三阶段框架可以很好地解决上述问题。首先，这三个网络有各自的任务，因此每个网络都可以快速收敛到自己的全局最优值。其次，这三个任务的相关性较弱，因此组装的框架更接近或更容易收敛到全局最优值。本文提出的任务分解框架可以比单级网络更好地处理复杂的夜间摄影渲染任务。  
        

效果展示：  
![image](https://github.com/user-attachments/assets/d97aa1e3-29b5-4654-a357-b8a8132da267)








       



（二）专用芯片：  
1.AI ISP目前在芯片层面主要采用的还是传统ISP+NPU的架构   



2.AI-ISP芯片：  

（1)安霸（Ambarella）：其CV（计算机视觉）系列芯片集成了先进的ISP功能与AI处理能力，广泛应用于安全摄像头、汽车应用和无人机。  

        Ambarella CV2x 5M SoC包括一个先进的CVflow计算机视觉处理引擎、一个1 GHz四核Arm Cortex-A53 CPU、一个5M H.264/H.265编码器、  
        一个带有Ambarella图像传感器处理器（ISP）的高性能数字信号处理器（DSP）子系统、提供3A、高动态范围（HDR）、去雾和3D降噪等ISP功能。  
        灵活的高清CV2x H.264-H.265编解码器可提供高达5Mp30 HEVC+1080p30 HEVC+5Mp4 MJPEG分辨率的记录，包括一个高质量、低延迟的二次流  
        CV2x支持多个传感器接口，可实现高达3200万像素分辨率的多种流行CMOS传感器。高达6.4亿像素/秒的输入速率。  
![image](https://github.com/user-attachments/assets/1ac14c87-28fc-4c6a-89fc-d22fb4e6cf65)
 
(2)Movidius Myriad X：  
![image](https://github.com/user-attachments/assets/a0a5a44a-ca68-4305-8308-d802447a1775)

        Movidius Myriad X 是英特尔公司开发的先进视觉处理单元（VPU），专为高性能、低功耗的计算机视觉应用而设计。  
        1.神经计算引擎：Myriad X 包含一个专用的神经计算引擎，这是一个专门用于深度学习推理的硬件加速器。该引擎每  
          秒可以执行高达 1 万亿次操作（TOPS），支持实时神经网络处理。  
        2.高性能成像和视觉处理：VPU 集成了 16 个可编程的 128 位 VLIW（超长指令字）矢量处理器，称为 SHAVE（流式混合架构矢量引擎）  
          核心，提供强大的并行处理能力。它可以同时支持多种视觉应用，例如物体检测、分类、分割和深度估计。  
        3.先进的图像处理：Myriad X 包含一个图像信号处理器（ISP），能够处理高级图像处理任务，如去噪、去畸变和高动态范围（HDR）成像。  
        4.低功耗：  Myriad X 的主要优势之一是其低功耗，使其适用于电池供电的设备，如无人机、AR/VR 头戴设备和智能相机。  
        5.集成和连接性：该芯片支持多种接口，包括 PCIe、USB 和 MIPI，便于与其他硬件组件和传感器集成。  
        6.3D 深度感知：支持通过立体深度实现 3D 深度感知，使其在需要空间感知的应用中表现出色，如机器人技术和增强现实。  



参考论文：  Model-Based Image Signal Processors via Learnable Dictionaries  
1.摘要：  

        1.数码相机通过其图像信号处理器（ISP）将传感器RAW读数转换为RGB图像。
        2.最近的方法试图通过估计RGB到RAW的映射来ISP：基于模型的手工方法是可解释和可控的，通常需要手动微调参数，  
          而端到端的可学习神经网络需要大量的训练数据，有时需要复杂的训练过程，并且通常缺乏可解释性和参数控制。  
        3.为了解决这些现有的局限性，我们提出了一种新的基于模型和数据驱动的混合ISP，该ISP基于规范的ISP操作，  
          既可学习又可解释。我们提出的可逆模型能够在RAW和RGB域之间进行双向映射，它采用了丰富参数表示的端到端学习，  
          即字典，不受直接参数监督，还可以进行简单合理的数据推理。  

2.网络结构图：  
![image](https://github.com/user-attachments/assets/b861990f-5e27-4811-b9fe-ef5174eeaf23)


