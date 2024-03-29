"""Wrappers around Pyserial instances"""
import numpy as np
from serial.serialutil import (
    PARITY_NONE, 
    PARITY_EVEN, 
    PARITY_ODD, 
    PARITY_MARK, 
    PARITY_SPACE
)

import serial

class BaseRotationStage:

    def __init__(self, port, baudrate, bytesize, data_bits, parity, stop_bits, termination_character, encoding):
        """init a rotation stage for motion control
        
        Notes:
        ------
        This was based on the required input for an AGILIS AG-UC8 motion controller, and
        communication via Pyserial. It is not tested for any other type of rotation stage,
        and may require additional inputs.

        Parameters
        ----------
        port : str
            port the motion controller is on, by Windows will use 'COMX',
            but linux will use something like '/dev/ttyusbXX'
        baudrate : int
            information transfer rate that motion controller communicates
            at (typically a set value)
        data_bits : int
            number of data bits (typically set by controller)
        parity : str
            method of detecting erorrs in transmission. Use PARITY_NAMES from Pyserial to 
            set this
        stop_bits : int
            number of bits after data transmitted
        termination_character : str
            What to use to denote line termination
        encoding : str
            What encoding to use to convert command strings to bytes
        """

        # set up instance serial communication
        self.serial_communication = serial.Serial(port=port,
                                                  baudrate=baudrate,
                                                  bytesize=bytesize,
                                                  parity=parity,
                                                  stopbits=stop_bits)
        
        self.termination_character = termination_character
        self.encoding = encoding

    def close(self):
        """close the serial communication
        """

        self.serial_communication.close()


class AgilisRotationStage(BaseRotationStage):

    def __init__(self, channel=1, axis=1, port='COM1', baudrate=921600, bytesize=8, data_bits=8, parity=PARITY_NONE, stop_bits=1, termination_character='\r\n', encoding='utf-8'):
        """_summary_

        Parameters
        ----------
        channel : int, optional
            Channel on the AG-UC2/8 controller, by default 1
        axis : int, optional
            Axis on the AG-UC2/8 controller, by default 1
        """
        super().__init__(port, baudrate, bytesize, data_bits, parity, stop_bits, termination_character, encoding)
        self.channel = channel
        self.axis = axis

    def set_step_delay(self, delay):
        """Sets the step delay of the stepping mode, applies for both positive and negative directions.
        Delay is programmed as multiples of 10 microseconds. Max value is equal to 2 second delay

        Parameters
        ----------
        delay : int
            delay in units of 10 microseconds
        """

        # set limits of step delay
        if delay < 0:
            delay = 0
        elif delay > 200000:
            delay = 200000

        commandstring = f'{self.axis} DL {delay}' + self.termination_character
        commandbytes = bytes(commandstring, encoding=self.encoding)
        self.serial_communication.write(commandbytes)

    def start_jog_motion(self, jog_mode):
        """Starts a jog motion at the defined speed specified by an integer

        Notes
        -----
        the sign of the jog mode defines forward (+) or backward (-) motion
        jog_mode = -4 or 4, 666 steps/s at defined step amplitude
        jog_mode = -3 or 3, 1700 steps/s at defined step amplitude
        jog_mode = -2 or 2, 100 steps/s at defined step amplitude
        jog_mode = -1 or 1, 5 steps/s at defined step amplitude
        jog_mode = 0, No move, go to READY state

        Parameters
        ----------
        jog_mode : _type_
            _description_
        """

        if np.abs(jog_mode) > 4:
            jog_mode = 0

        commandstring = f'{self.axis} JA {jog_mode}' + self.termination_character
        commandbytes = bytes(commandstring, encoding=self.encoding)
        self.serial_communication.write(commandbytes)

    def tell_limit_status(self):
        """Returns the limits switch status of the controller, the returns are the following

        PH0 : No limit switch active
        PH1 : Limit switch of channel #1 is active, #2 is not
        PH2 : Limit switch of channel #2 is active, #1 is not
        PH3 : Limit switch of channel #1 and #2 are
        """

        commandstring = 'PH'
        commandbytes = bytes(commandstring, encoding=self.encoding)
        self.serial_communication.write(commandbytes)

    def set_mode_local(self):
        """sets to local mode, where only status queries are allowed
        """

        commandstring = 'ML' + self.termination_character
        commandbytes = bytes(commandstring, encoding=self.encoding)
        self.serial_communication.write(commandbytes)

    def set_mode_remote(self):
        """sets to local mode, where status queries and motion control are allowed
        """

        commandstring = 'MR' + self.termination_character
        commandbytes = bytes(commandstring, encoding=self.encoding)
        self.serial_communication.write(commandbytes)

    def set_step_amplitude(self, amplitude):
        """Sets step amplitude in positive and negative direction.

        Notes
        -----
        Only positive integers are permitted, so amplitude is automatically converted to
        its nearest integer magnitude

        Too small a step size may result in nonlinear motion, or no motion at all. Furthermore
        there is no gurantee of a linear correlation between the step amplitude and effective 
        motion size, so it must be calibrated before experimentation.

        Parameters
        ----------
        amplitude : int
            amplitude of steps taken
        """

        amplitude = int(np.abs(amplitude))
        if amplitude > 50:
            amplitude = 50
        
        commandstring = f'{self.axis} SU {amplitude}' + self.termination_character
        commandbytes = bytes(commandstring, encoding=self.encoding)
        self.serial_communication.write(commandbytes)

    def relative_move(self, steps):
        """Move relative to current position in steps whose amplitude are defined by the SU command
        (defaults to 16)

        Parameters
        ----------
        steps : int
            number of steps to move
        """

        commandstring = f'{self.axis} PR {steps}' + self.termination_character
        commandbytes = bytes(commandstring, encoding=self.encoding)
        self.serial_communication.write(commandbytes)

    def absolute_move(self, target_position):
        """starts a process to move to an absolute position, this interrupts USB communication. 
        After the position is found USB coms are opened again.

        Notes
        -----
        The absolute position is found by moving from the current position to the motion limit, twice
        As such, this takes some time and is not reccomended for quick use.

        Parameters
        ----------
        target_position : int
            target position in 1/1000th of total travel
        """

        commandstring = f'{self.axis} PA {target_position}' + self.termination_character
        commandbytes = bytes(commandstring, encoding=self.encoding)
        self.serial_communication.write(commandbytes)

    def zero_position(self):
        """resets the step counter to zero
        """

        commandstring = f'{self.axis} ZP' + self.termination_character
        commandbytes = bytes(commandstring, encoding=self.encoding)
        self.serial_communication.write(commandbytes)

    




