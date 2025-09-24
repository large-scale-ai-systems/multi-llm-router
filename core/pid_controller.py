"""
PID Controller implementation for adaptive LLM routing.

This module implements a Proportional-Integral-Derivative controller
for dynamic allocation adjustment based on performance metrics.
"""

import time
import numpy as np
from typing import Optional, Tuple


class PIDController:
    """PID Controller for adaptive LLM routing"""
    
    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.01, 
                 target: float = 0.0, output_limits: Tuple[float, float] = (-1.0, 1.0)):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain  
        self.kd = kd  # Derivative gain
        self.target = target
        self.output_limits = output_limits
        
        # Internal state
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()
        
    def update(self, current_value: float, dt: Optional[float] = None) -> float:
        """Update PID controller and return control output"""
        current_time = time.time()
        if dt is None:
            dt = current_time - self.prev_time
        
        if dt <= 0.0:
            dt = 0.01  # Prevent division by zero
            
        # Calculate error
        error = self.target - current_value
        
        # Proportional term
        # Proportional (P) - Immediate Response
        # Purpose: Responds directly to current performance deviation
        # Example: If GPT-4 is performing 20% below target, immediately reduce its allocation
        # Benefit: Quick corrections to obvious performance issues
        proportional = self.kp * error
        
        # Integral (I) - Historical Correction
        # Purpose: Addresses accumulated performance issues over time
        # Example: If Claude consistently underperforms by small amounts, gradually shift traffic away
        # Benefit: Eliminates steady-state errors that P alone can't fix        
        self.integral += error * dt
        self.integral = np.clip(self.integral, -10.0, 10.0)  # Anti-windup protection
        integral = self.ki * self.integral
        
        # Derivative (D) - Predictive Response
        # Purpose: Predicts performance trends and responds before they worsen
        # Example: If Gemini's performance is rapidly declining, preemptively reduce allocation
        # Benefit: Prevents overshoot and oscillation
        derivative = self.kd * (error - self.prev_error) / dt
        
        # Calculate output
        output = proportional + integral + derivative
        output = np.clip(output, self.output_limits[0], self.output_limits[1]) #Max X% allocation change
        
        # Update state
        self.prev_error = error
        self.prev_time = current_time
        
        return output