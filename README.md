# MPRNet Image Restoration Setup Guide

## 1. Start the Docker Environment
Run this command from the project root directory to build the image and start the container in the background:

```bash
docker-compose up -d --build

```

---

## 2. Navigate to the Workspace

Attach to the container's interactive terminal. This places you directly inside the `/workspace/MPRNet` directory:

```bash
docker exec -it mprnet_submission bash

```

---

## 3. Running Scripts

Navigate to the respective task module directory to run evaluation or demonstration scripts.

### Deblurring Module

```bash
cd /workspace/MPRNet/Deblurring

```

* **Evaluate on Test Dataset:**
```bash
python evaluate_deblurring.py

```


* **Run Visual Demonstration:**
```bash
python GoPro_demo_deblurring.py

```



### Denoising Module

```bash
cd /workspace/MPRNet/Denoising

```

* **Evaluate on Test Dataset:**
```bash
python evaluate_denoising.py

```


* **Run Visual Demonstration:**
```bash
python demo_denoising.py

```



### Deraining Module

```bash
cd /workspace/MPRNet/Deraining

```

* **Evaluate on Test Dataset:**
```bash
python evaluate_deraining.py

```


* **Run Visual Demonstration:**
```bash
python demo_deraining.py

```



---

## 4. Environment Teardown

To exit the container interactive shell and shut down the active services:

```bash
# Inside container terminal
exit

# On host machine terminal
docker-compose down

```

```

```