/*------------------------------------------
 FILE:    digital_input.c

DESCRIPTION: Demo of how to read a digital input. 
PICmicro streams the value of the input pin to the PC. 
-------------------------------------------*/

#include "mcc_generated_files/mcc.h"

#define LED      RC1   // LED 
#define BUTTON   RC0   // switch

void putch(char txData); // declare eusart function

/*******************************************************
  Main starts here
*******************************************************/

void main(void)
{
  char i, btn;

//-------initialize the device
  SYSTEM_Initialize();

//-------Flash led a few times
  for (i=0;i<3;i++) {
    LED=1;
    __delay_ms(100);
    LED=0;
   __delay_ms(300);
  }

//------Setup PC terminal using VT100 control codes
  putch(27); putch('c');  //reset terminal
  __delay_ms(10);
  putch(27); putch('H');  //cursor to home position
  __delay_ms(10);

  printf("\rButton value:\r\n");

//-------Endless loop printing the value of the input
  while (1) {
    __delay_ms(100);     // about 10 Hz sampling rate
    btn = BUTTON;  //read the button
    LED != BUTTON;
    printf("\r  %2d",btn);
  }
}