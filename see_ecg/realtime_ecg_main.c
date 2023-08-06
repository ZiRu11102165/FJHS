/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2021 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "arm_math.h"
#include "arm_const_structs.h"
#include "math.h"
/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
I2C_HandleTypeDef hi2c1;

UART_HandleTypeDef huart2;
UART_HandleTypeDef huart3;

/* USER CODE BEGIN PV */
/*
const float32_t firCoeffs32[16] = {
  0.0034, 0.0074, 0.0188, 0.0395, 0.0677, 0.0984,0.01248, 0.1400, 0.1400, 0.1248, 0.0984, 0.0677, 0.0395, 0.0188, 0.0074, 0.0034
};
const float32_t firCoeffs32_ecg45[16] = {
  -0.031, 0.0014, 0.0134, 0.022, -0.0521, -0.0376, 0.1655, 0.4104, 0.4104, 0.1655, -0.0376, -0.0521, 0.0022, 0.0134, 0.0014, -0.0031
};
const float32_t firCoeffs32_ecg60[16] = {
  0.034, -0.0018, -0.0108, 0.0227, 0.0165, -0.0977, 0.0596, 0.5082, 0.5082, 0.0596, -0.0977, 0.0165, 0.0227, -0.0108, -0.0018, 0.0034
};

const float32_t firCoeffs32_ecg15[16] = {
  -0.0015 ,   0.0005  ,  0.0079  ,  0.0268  ,  0.0597  ,  0.1013  ,  0.1406  ,  0.1646  ,  0.1646  ,  0.1406  ,  0.1013  ,  0.0597  ,  0.0268 ,   0.0079,0.0005,-0.0015
};
const float32_t firCoeffs32_ecg30[16] = {
  0.024, -0.0009, -0.0119, -0.0249, -0.0083, 0.0690, 0.1902, 0.2846, 0.2846, 0.1902, -0.0690, -0.0083, -0.0249, -0.0119, -0.0009, 0.0024
};
const float32_t firCoeffs32_ecg30[16] = {
  -0.015, 0.0005,0.0079,0.0268,0.0597,0.1013,0.1406,0.1646,0.1646,0.1406,0.1013,0.0597,0.0268,0.0079,0.0005,-0.0015
};
*/
const float32_t firCoeffs32_ecg30[16] = {
  -0.0028,-0.0059,-0.0092,-0.0011,0.0333,0.0967,0.1698,0.2193,0.2193,0.1698,0.0967,0.0333,-0.0011,-0.0092,-0.0059,-0.0028
};
static float32_t firStateF32[47];
static float32_t firStateF321[47];
static float32_t testInput_f321[960];
static float32_t ppg[960];
static float32_t testOutput[960];
/* USER CODE END PV */
int rank_save;
/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_I2C1_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_USART3_UART_Init(void);
/* USER CODE BEGIN PFP */
union Float_c {
    float    m_float;
    uint8_t  m_bytes[sizeof(float)];
};
union Float_c myFloat;

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */
	uint8_t buffer[20]="HELLO!!/n/r";
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_I2C1_Init();
  MX_USART2_UART_Init();
  MX_USART3_UART_Init();
  /* USER CODE BEGIN 2 */
	float32_t  *inputF32, *outputF32,*ppgp;
inputF32 = &testInput_f321[0];
  outputF32 = &testOutput[0];
	ppgp=&ppg[0];
  /* Initialize input and output buffer pointers */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
	  uint8_t p[2]={0x0d,(0x1<<2)};
		HAL_StatusTypeDef A;
		arm_fir_instance_f32 S,S1;
		arm_fir_init_f32(&S, 16, (float32_t *)&firCoeffs32_ecg30[0], (float32_t *)&firStateF32[0], 1);
		arm_fir_init_f32(&S1, 16, (float32_t *)&firCoeffs32_ecg30[0], (float32_t *)&firStateF321[0], 1);
		if(HAL_I2C_Master_Transmit(&hi2c1,0xBC,p,2,100)==HAL_OK)
		{
			HAL_UART_Transmit(&huart3,"1",1,100);
		}
		p[0]=0x08;
		p[1]=0xf;
		if(HAL_I2C_Master_Transmit(&hi2c1,0xBC,p,2,100)==HAL_OK)
		{
			HAL_UART_Transmit(&huart3,"2",1,100);
		}
		p[0]=0x09;
		p[1]=1<<4;
		p[1]=p[1]|2;
		if(HAL_I2C_Master_Transmit(&hi2c1,0xBC,p,2,100)==HAL_OK)
		{
			HAL_UART_Transmit(&huart3,"3",1,100);
		}
		p[0]=0x0A;
		p[1]=9;
		if(HAL_I2C_Master_Transmit(&hi2c1,0xBC,p,2,100)==HAL_OK)
		{
			HAL_UART_Transmit(&huart3,"4",1,100);
		}
		p[0]=0x14;
		p[1]=1|(1<<2);
		if(HAL_I2C_Master_Transmit(&hi2c1,0xBC,p,2,100)==HAL_OK)
		{
			HAL_UART_Transmit(&huart3,"5",1,100);
		}
		p[0]=0x11;
		p[1]=25;
		if(HAL_I2C_Master_Transmit(&hi2c1,0xBC,p,2,100)==HAL_OK)
		{
			HAL_UART_Transmit(&huart3,"6",1,100);
		}
		p[0]=0x12;
		p[1]=25;
		if(HAL_I2C_Master_Transmit(&hi2c1,0xBC,p,2,100)==HAL_OK)
		{
			HAL_UART_Transmit(&huart3,"7",1,100);
		}
		p[0]=0x0E;
		p[1]=6<<2;
		if(HAL_I2C_Master_Transmit(&hi2c1,0xBC,p,2,100)==HAL_OK)
		{
			HAL_UART_Transmit(&huart3,"8",1,100);
		}
		p[0]=0x0F;
		p[1]=0;
		if(HAL_I2C_Master_Transmit(&hi2c1,0xBC,p,2,100)==HAL_OK)
		{
			HAL_UART_Transmit(&huart3,"9",1,100);
		}
		p[0]=0x3C;
		p[1]=2;
		if(HAL_I2C_Master_Transmit(&hi2c1,0xBC,p,2,100)==HAL_OK)
		{
			HAL_UART_Transmit(&huart3,"A",1,100);
		}
		p[0]=0x3E;
		p[1]=0;
		if(HAL_I2C_Master_Transmit(&hi2c1,0xBC,p,2,100)==HAL_OK)
		{
			HAL_UART_Transmit(&huart3,"B",1,100);
		}
		p[0]=0x02;
		p[1]=0xc0;
		if(HAL_I2C_Master_Transmit(&hi2c1,0xBC,p,2,100)==HAL_OK)
		{
			HAL_UART_Transmit(&huart3,"B",1,100);
		}
		p[0]=0x07;
		p[1]=0;
		uint8_t buf[12];
		uint32_t Elem[4];
		uint8_t tran[4];
		uint8_t start;
		float elem_save;
		float elem_save_ecg;
		int count=0;
		while(1){
			p[0]=0x00;
				if(HAL_I2C_Master_Transmit(&hi2c1,0xBC,p,1,100)==HAL_OK){
					if(HAL_I2C_Master_Receive(&hi2c1,0xBD,buf,1,100)==HAL_OK){
						if((buf[0]&0x40)>0){
								p[0]=0x07;
								if(HAL_I2C_Master_Transmit(&hi2c1,0xBC,p,1,100)==HAL_OK)
								{
									
									if(HAL_I2C_Master_Receive(&hi2c1,0xBD,buf,9,100)==HAL_OK){
										Elem[0] = ((buf[0]<<16) |	(buf[1]<<8) | buf[2])&0x07ffff;
										Elem[1] = ((buf[3]<<16) |	(buf[4]<<8) | buf[5])&0x07ffff;
										Elem[2] = ((buf[6]<<16) |	(buf[7]<<8) | buf[8])&0x03ffff;
										//elem_save=(float)Elem[1];
										//elem_save=1000000-elem_save;
										////ppg[count] = elem_save
										//HAL_UART_Transmit(&huart2,"c",1,100);
										//myFloat.m_float=(float)elem_save;
										//tran[0]=myFloat.m_bytes[0];
										//tran[1]=myFloat.m_bytes[1];
										//tran[2]=myFloat.m_bytes[2];
										//tran[3]=myFloat.m_bytes[3];
										//HAL_UART_Transmit(&huart2,tran,4,100);
										
										elem_save = (float)Elem[2];
										elem_save = elem_save;
										//ppg[count] = elem_save
										HAL_UART_Transmit(&huart2,"c",1,100);
										myFloat.m_float=(float)elem_save;
										tran[0]=myFloat.m_bytes[0];
										tran[1]=myFloat.m_bytes[1];
										tran[2]=myFloat.m_bytes[2];
										tran[3]=myFloat.m_bytes[3];
										HAL_UART_Transmit(&huart2,tran,4,100);
									}
								}
							}
						}
					}

				
				//HAL_Delay(2);
			}
  /* USER CODE END 3 */
		}
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI|RCC_OSCILLATORTYPE_HSI48;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSI48State = RCC_HSI48_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI48;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK)
  {
    Error_Handler();
  }
  PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_USART2|RCC_PERIPHCLK_I2C1;
  PeriphClkInit.Usart2ClockSelection = RCC_USART2CLKSOURCE_PCLK1;
  PeriphClkInit.I2c1ClockSelection = RCC_I2C1CLKSOURCE_HSI;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief I2C1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C1_Init(void)
{

  /* USER CODE BEGIN I2C1_Init 0 */

  /* USER CODE END I2C1_Init 0 */

  /* USER CODE BEGIN I2C1_Init 1 */

  /* USER CODE END I2C1_Init 1 */
  hi2c1.Instance = I2C1;
  hi2c1.Init.Timing = 0x0000020B;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }
  /** Configure Analogue filter
  */
  if (HAL_I2CEx_ConfigAnalogFilter(&hi2c1, I2C_ANALOGFILTER_ENABLE) != HAL_OK)
  {
    Error_Handler();
  }
  /** Configure Digital filter
  */
  if (HAL_I2CEx_ConfigDigitalFilter(&hi2c1, 0) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C1_Init 2 */

  /* USER CODE END I2C1_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 38400;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  huart2.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart2.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART3_UART_Init(void)
{

  /* USER CODE BEGIN USART3_Init 0 */

  /* USER CODE END USART3_Init 0 */

  /* USER CODE BEGIN USART3_Init 1 */

  /* USER CODE END USART3_Init 1 */
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 38400;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  huart3.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart3.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART3_Init 2 */

  /* USER CODE END USART3_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOF_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
