"""Wrappers around Pyserial instances"""
import numpy as np # uses true numpy

from serial.serialutil import (
    PARITY_NONE, 
    PARITY_EVEN, 
    PARITY_ODD, 
    PARITY_MARK, 
    PARITY_SPACE
)

import serial

class BaseRotationStage:

    def __init__(self, port, baudrate, bytesize, data_bits, parity, stop_bits, termination_character, encoding, timeout):
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
        timeout : float
            Set read timeout value in seconds
        """

        # set up instance serial communication
        self.serial_communication = serial.Serial(port=port,
                                                  baudrate=baudrate,
                                                  bytesize=bytesize,
                                                  parity=parity,
                                                  stopbits=stop_bits,
                                                  timeout=timeout)
        
        self.termination_character = termination_character
        self.encoding = encoding

    def close(self):
        """close the serial communication
        """

        self.serial_communication.close()


class AgilisRotationStage(BaseRotationStage):

    def __init__(self, channel=1, axis=1, port='COM1', baudrate=921600, bytesize=8, data_bits=8, parity=PARITY_NONE, stop_bits=1, timeout=1, termination_character='\r\n', encoding='utf-8'):
        """_summary_

        Parameters
        ----------
        channel : int, optional
            Channel on the AG-UC2/8 controller, by default 1
        axis : int, optional
            Axis on the AG-UC2/8 controller, by default 1

        Notes
        -----
        Inherits from BaseRotationStage
        """
        super().__init__(port, baudrate, bytesize, data_bits, parity, stop_bits, termination_character, encoding, timeout)
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

    def get_step_delay(self):
        """return current step delay

        FIXME: This stalls indefinitely, can't figure out source. It might be
        that it's waiting for 8 bytes and not returning 8 bytes?

        """

        commandstring = f'{self.axis} DL?' + self.termination_character
        commandbytes = bytes(commandstring, encoding=self.encoding)
        self.serial_communication.write(commandbytes)
        out = self.serial_communication.readline()
        
        return out

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
        jog_mode : int
            jog mode to begin motion
        """

        if np.abs(jog_mode) > 4:
            jog_mode = 0

        commandstring = f'{self.axis} JA {jog_mode}' + self.termination_character
        commandbytes = bytes(commandstring, encoding=self.encoding)
        self.serial_communication.write(commandbytes)

    def get_jog_mode(self):
        """return current jog mode, see start_jog_motion
        """

        commandstring = f'{self.axis} JA?' + self.termination_character
        commandbytes = bytes(commandstring, encoding=self.encoding)
        self.serial_communication.write(commandbytes)
        out = self.serial_communication.readline()

        return out
    
    def measure_current_position(self):
        """starts a process to measure current position. Interrupts USB communication 
        during the process. This can last up to 2 minutes
        """

        commandstring = f'{self.axis} MA' + self.termination_character
        commandbytes = bytes(commandstring, encoding=self.encoding)
        self.serial_communication.write(commandbytes)
        out = self.serial_communication.readline()

        return out


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

    def move_to_limit(self, jog_mode):
        """Moves to limit with a jog motion at the defined speed specified by an integer

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
        jog_mode : int
            jog mode to begin motion
        
        """

        commandstring = f'{self.axis} MV {jog_mode}' + self.termination_character
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


    def get_current_target_position(self):
        """Returns current target position when using absolute move
        """

        commandstring = f'{self.axis} PA?' + self.termination_character
        commandbytes = bytes(commandstring, encoding=self.encoding)
        self.serial_communication.write(commandbytes)
        out = self.serial_communication.readline()

        return out
    

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

    def reset_controller(self):
        """Resets the controller, all temporary settings are reset to default and 
        controller is in local mode
        """

        commandstring = 'RS'
        commandbytes = bytes(commandstring, encoding=self.encoding)
        self.serial_communication.write(commandbytes)

    def stop_motion(self):
        """Stops the motion on the defined axis, sets state to be ready
        """
        
        commandstring = f'{self.axis} ST' + self.termination_character
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

    
    def get_step_amplitude(self):
        """Returns the current step amplitude, should be in the range -50 to 50
        """

        commandstring = f'{self.axis} SU?' + self.termination_character
        commandbytes = bytes(commandstring, encoding=self.encoding)
        self.serial_communication.write(commandbytes)
        out = self.serial_communication.readline()

        return out
    
    
    def get_previous_command_error(self):
        """get the error of the previous command,

        FIXME: This stalls indefinitely if there's no error

        Error Codes
        -------
        0 : No Error
        -1 : Unknown command
        -2 : Axis out of range
        -3 : Wrong format for parameter nn
        -4 : Parameter nn out of range
        -5 : Not allowed in local mode
        -6 : Note allowed in current state
        """

        commandstring = 'TE'
        commandbytes = bytes(commandstring, encoding=self.encoding)
        self.serial_communication.write(commandbytes)
        out = self.serial_communication.readline()

        return out
    

    def get_number_of_steps(self):
        """Returns the number of accumullated steps in the forward direction minus the
        number of steps in backward direction since powering the controller or since
        the last zero point.
        """

        commandstring = f'{self.axis} TP' + self.termination_character
        commandbytes = bytes(commandstring, encoding=self.encoding)
        self.serial_communication.write(commandbytes)
        out = self.serial_communication.readline()
        
        return out
    
    def get_axis_status(self):
        """Returns the status of the axis

        FIXME: This stalls indefinitely when not moving

        Axis Status
        -------
        0 : Ready (Not moving)
        1 : Stepping (Currently executing a PR command)
        2 : Jogging (Currently executing a JA command)
        3 : Moving to limit (Currently executing a MV, MA, or PA command)
        """

        commandstring = f'{self.axis} TS' + self.termination_character
        commandbytes = bytes(commandstring, encoding=self.encoding)
        self.serial_communication.write(commandbytes)
        out = self.serial_communication.readline()

        return out


    def zero_position(self):
        """resets the step counter to zero
        """

        commandstring = f'{self.axis} ZP' + self.termination_character
        commandbytes = bytes(commandstring, encoding=self.encoding)
        self.serial_communication.write(commandbytes)

    @property
    def angular_step_size(self):
        return self._angular_step_size

    @angular_step_size.setter
    def angular_step_size(self, value):
        self._angular_step_size = value

    @property
    def angular_offset(self):
        return self._angular_offset
    
    @angular_offset.setter
    def angular_offset(self, value):
        self.angular_offset = value
    
    def compute_angular_position(self):
        outstring = self.get_number_of_steps().decode(self.encoding)
        parsed_outstring = outstring[4:-2] # start after TP, end before termination character
        steps = int(parsed_outstring)
        self.angular_position = steps * self.angular_step_size

    @property
    def angular_position(self):
        return self._angular_position
    
    @angular_position.setter
    def angular_position(self, value):
        self._angular_position = value


    




