class PIDController:
    """
    Robust Proportional-Integral-Derivative (PID) Controller.
    Includes anti-windup clamping, derivative filtering, and limit constraints.
    """
    def __init__(self, kp, ki, kd, output_min=None, output_max=None, anti_windup=True, derivative_filter_coeff=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.output_min = output_min
        self.output_max = output_max
        self.anti_windup = anti_windup
        self.filter_coeff = derivative_filter_coeff  # Alpha for low-pass filtering of derivative term

        self.prev_error = 0.0
        self.integral = 0.0
        self.filtered_derivative = 0.0

    def compute(self, setpoint, measurement, dt):
        if dt <= 0:
            return 0.0

        # Calculate error
        error = setpoint - measurement

        # Proportional term
        p_term = self.kp * error

        # Integral term with optional anti-windup clamping
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Raw derivative
        raw_derivative = (error - self.prev_error) / dt
        
        # Low-pass filter the derivative term to suppress high-frequency noise
        self.filtered_derivative = (self.filter_coeff * raw_derivative) + ((1.0 - self.filter_coeff) * self.filtered_derivative)
        d_term = self.kd * self.filtered_derivative

        # Calculate raw control effort
        output = p_term + i_term + d_term

        # Clamp output to limits and apply anti-windup
        if self.output_min is not None or self.output_max is not None:
            clamped_output = output
            if self.output_min is not None and output < self.output_min:
                clamped_output = self.output_min
            elif self.output_max is not None and output > self.output_max:
                clamped_output = self.output_max

            # Anti-windup clamping: if saturated and error is in same direction, back out the integration
            if self.anti_windup and (clamped_output != output):
                # If error sign matches output saturation direction, prevent accumulation
                if (error > 0 and output > 0) or (error < 0 and output < 0):
                    self.integral -= error * dt  # Revert integration step
                    i_term = self.ki * self.integral
                    output = p_term + i_term + d_term
                    
            output = clamped_output

        self.prev_error = error
        return output

    def reset(self):
        """Resets the controller state"""
        self.prev_error = 0.0
        self.integral = 0.0
        self.filtered_derivative = 0.0
