import serial

def colorize(text, color):
    """
    Display text with some ANSI color in the terminal.
    """
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])

def bold(text):
    """
    Display text in bold in the terminal.
    """
    return colorize(text, "1")

def available_serial_ports():
    ports = ['COM%s' % (i + 1) for i in range(10)]
    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result

def uart_capture(callback_func=None,allow_prints=True):
    """The RTK procotol used to output raw data by UART is as follows:
        
        * Input check
            - All data is little endian
                
                        A55101004000A80200
            
            - A551: pairing word
            - 0100: number of streams (1)
            - 4000: number of samples per message (64 or 0x0040)
            - A80200: sequency counter 

        * Raw data

                FFFF100018000700030012000B001F000F000500F3FF0....

            - The data is 16 bit Q15 little endian with the same size set
            in the DSP code (FrameDuration)

        * ACK final words

                                7301B6

            - 7301: checksum value
            - B6: message end byte

        Here is an example of the the total message:

        A55101004000A80200 (9 bytes)
        FBFF050008000B001300070005... (128 bytes)
        7301B6 (3 bytes)

        UART caracteristics:
        - Baud rate 2000000 (2M)
        - Pairity None
        - Data Bits 8
        - Stop bits 1
        
        Inputs:
            - callback_func: the function called when a new frame is available.
                            The input data is a Python size 64 list of floats
            - allow_prints: True to print code status 

        Notes:
        - This code only works with frame size of 64 and one stream

        ------
        from real_time_uart_capture import *

        def my_callback_func(frame):
            print(frame)

        uart_capture(callback_func=my_callback_func,allow_prints=True)
        ----
    """
    if allow_prints:
        print("\n--------------------   RTK 8773CO realtime DSP UART capture   -----------------------")
	    
    if callback_func is None:
        if allow_prints: print("- No callback function entered!!!")
        return None

    ports=""
    av_ports=available_serial_ports()
    for i in av_ports: 
        ports=ports+str(i)+","
    if(allow_prints): print("- Ports avaible:",ports[:-1])

    if(len(av_ports)==0): 
        if(allow_prints): print("- No avaible ports!!!")
        return None

    if(allow_prints): print("- Connecting to port",av_ports[0],"...")	

    ser = serial.Serial(port=av_ports[0], baudrate=2000000, bytesize=8, timeout=1, stopbits=serial.STOPBITS_ONE, parity= serial.PARITY_NONE)

    if ser.isOpen():
        if(allow_prints): print("- Connection successful!!!")

    raw_data=""
    counter=0
    if(allow_prints): print("- Starting data sync ...")
    while True:
        #Use the first message to sync data (first message data discarted)
        data=ser.read()
        raw_data=raw_data+data.hex()
        sync_word=raw_data[-6:]
        #Checking if sync word is in the buffer
        if(sync_word=="b6a551"):
            if(allow_prints): print("- Data sync successful!!!")
            #Start collecting framedata based on message size
            if(allow_prints): print("- Collecting input frames data ...")
            while True:
                counter=counter+1
                #Number of chars per message (Protocol + data)
                data=ser.read(140)
                raw_data=data.hex()
                #String has 2 elements per byte -> 140 bytes == 280 string chars
                if(raw_data[:8]!="01004000" or len(raw_data)!=280): 
                    if(allow_prints): print("- Data sync lost!!!")
                    return None
                raw_frame=raw_data[14:-10]
                #Input is big endian
                # frame=[int(raw_frame[i*4:(4+(i*4))],16) for i in range(64)]
                #Input is little endian
                frame=[int(raw_frame[(i*4)+2:(4+(i*4))]+raw_frame[i*4:(2+(i*4))],16) for i in range(64)]
                output_frame=[]
                for i in frame:
                    if(i<2**15): output_frame.append((i/2**15))
                    else: output_frame.append((i/2**15)-2)
                #Passing the new frame to the callback function
                callback_func(output_frame)
                

