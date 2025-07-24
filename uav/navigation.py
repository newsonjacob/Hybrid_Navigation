# uav/navigation.py
"""Navigation utilities for issuing motion commands to an AirSim drone."""
import time
import math
import logging
import airsim
from uav import config
from uav.navigation_state import NavigationState

logger = logging.getLogger(__name__)


class Navigator:
    """High level movement controller for an AirSim drone."""

    def __init__(self, client):
        """Create a new ``Navigator``.

        Parameters
        ----------
        client : airsim.MultirotorClient
            AirSim client instance used to send movement commands.
        """

        self.client = client
        self.braked = False
        self.dodging = False
        self.last_movement_time = time.time()
        self.grace_used = False  # add in __init__
        self.grace_period_end_time: float = 0.0
        self.just_resumed = False
        self.resume_grace_end_time = 0

        # Add obstacle detection hysteresis tracking
        self.obstacle_detection_count = 0
        self.obstacle_clear_count = 0
        self.obstacle_confirmed = False
        self.DETECTION_THRESHOLD = 1  # Frames required to confirm detection/clearing
        self.CLEAR_THRESHOLD = 2  # Frames required to confirm clearing
        
        # Store individual condition states for logging
        self.last_sudden_rise = False
        self.last_center_blocked = False
        self.last_combination_flow = False
        self.last_minimum_flow = False

        # Add max dodge duration tracking
        self.MAX_DODGE_DURATION = 5.0  # Maximum seconds to dodge before forcing resume
        self.dodge_start_time = None   # Track when dodge started
        self.dodge_direction = None    # Current dodge direction
        self.dodge_strength = 1.0   # Current dodge strength

        # Add post-dodge grace period for combination_flow
        self.POST_DODGE_GRACE_DURATION = 2.0  # Grace period after dodge completes
        self.post_dodge_grace_end_time = 0.0   # When grace period ends
        self.in_post_dodge_grace = False       # Flag for grace period state

    def get_state(self):
        """Return the drone position, yaw angle and speed.

        Returns
        -------
        Tuple[airsim.Vector3r, float, float]
            Position vector, yaw in degrees and speed magnitude in m/s.
        """
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        ori = state.kinematics_estimated.orientation
        yaw = math.degrees(airsim.to_eularian_angles(ori)[2])
        vel = state.kinematics_estimated.linear_velocity
        speed = math.sqrt(vel.x_val ** 2 + vel.y_val ** 2 + vel.z_val ** 2)
        return pos, yaw, speed

    def brake(self):
        """Immediately halt forward motion.

        Sends a short reverse velocity command proportional to the current
        forward speed.

        Returns
        -------
        NavigationState
            ``NavigationState.BRAKE`` for logging.
        """
        try:
            # Get current velocity and position
            state = self.client.getMultirotorState()
            vel = state.kinematics_estimated.linear_velocity
            current_z = state.kinematics_estimated.position.z_val  # Get current altitude
            
            speed = vel.x_val
            # Apply reverse velocity proportional to current forward speed (clamp if needed)
            reverse_speed = -min(speed, 3.0)  # Limit max reverse speed for safety
            
            # FIX: Use moveByVelocityZAsync to maintain altitude
            self.client.moveByVelocityZAsync(
                reverse_speed,  # X velocity (reverse)
                0,             # Y velocity (no lateral movement)
                current_z,     # Z position (maintain current altitude)
                0.5            # Duration
            )
        except AttributeError:
            # Fallback if state unavailable
            state = self.client.getMultirotorState()
            current_z = state.kinematics_estimated.position.z_val
            
            self.client.moveByVelocityZAsync(0, 0, current_z, 0.5)
            
        self.braked = True
        return NavigationState.BRAKE

    def dodge(self, smooth_L, smooth_C, smooth_R, direction=None, duration=2.0):
        """Sidestep left or right to avoid an obstacle.

        Parameters
        ----------
        smooth_L, smooth_C, smooth_R : float
            Smoothed optical flow magnitudes used for logging.
        direction : {"left", "right"}, optional
            Side to dodge toward. ``None`` defaults to left.
        duration : float, optional
            How long to apply the dodge command in seconds. Defaults to ``2.0``.

        Returns
        -------
        NavigationState
            ``NavigationState.DODGE_LEFT`` or ``NavigationState.DODGE_RIGHT``.
        """
        
        # Record dodge start time if not already dodging
        if not self.dodging:
            self.dodge_start_time = time.time()
            logger.info(f"[DODGE] Starting {direction or 'left'} dodge at {self.dodge_start_time:.2f}")
        
        # Set direction (default to left if None)
        if direction is None:
            direction = "left"
        
        lateral = 1.0 if direction == "right" else -1.0
        strength = 0.5 # Default dodge strength
        forward_speed = 0.0

        # Stop before dodging
        self.brake()
        time.sleep(0.5)  # Allow time for braking to take effect
        
        # FIX: Get current altitude and maintain it during dodge
        state = self.client.getMultirotorState()
        current_z = state.kinematics_estimated.position.z_val  # NED: negative up

        # Execute dodge movement
        self.client.moveByVelocityBodyFrameAsync(forward_speed, lateral * strength, current_z, duration)

        # Set dodge state
        self.dodging = True
        self.braked = False
        self.last_movement_time = time.time()
        self.dodge_direction = direction
        self.dodge_strength = strength
        
        return NavigationState.DODGE_RIGHT if direction == "right" else NavigationState.DODGE_LEFT

    def maintain_dodge(self):
        """Continue the dodge while an obstacle is still detected."""
        if self.dodging:
            lateral = 1.0 if self.dodge_direction == "right" else -1.0
            self.client.moveByVelocityBodyFrameAsync(0.0, lateral * self.dodge_strength, 0, duration=0.3)

    def resume_forward(self):
        """Resume normal forward flight after braking or dodging.

        Returns
        -------
        NavigationState
            ``NavigationState.RESUME``.
        """

        # Log dodge duration if we were dodging
        if self.dodging and self.dodge_start_time is not None:
            dodge_duration = time.time() - self.dodge_start_time
            logger.info(f"[DODGE] Completed after {dodge_duration:.2f}s")

            # Start post-dodge grace period
            self.post_dodge_grace_end_time = time.time() + self.POST_DODGE_GRACE_DURATION
            self.in_post_dodge_grace = True
            logger.info(f"[GRACE] Post-dodge grace period started for {self.POST_DODGE_GRACE_DURATION}s")

        # Stop before resuming forward motion
        self.client.moveByVelocityAsync(0, 0, 0, 0)
        time.sleep(0.2)  # Allow time for braking to take effect
        
        state = self.client.getMultirotorState()
        z = state.kinematics_estimated.position.z_val  # NED: z is negative up
        self.client.moveByVelocityZAsync(2, 0, z, duration=3,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(False, 0))
        
        #Reset dodge state
        self.braked = False
        self.dodging = False
        self.dodge_start_time = None  # Reset dodge timing
        self.dodge_direction = None
        self.just_resumed = True
        self.resume_grace_end_time = time.time() + 0 # 0 second grace
        self.last_movement_time = time.time()
        return NavigationState.RESUME

    def check_post_dodge_grace(self):
        """Check and update post-dodge grace period status.
        
        Returns
        -------
        bool
            True if currently in post-dodge grace period.
        """
        current_time = time.time()
        
        if self.in_post_dodge_grace:
            if current_time >= self.post_dodge_grace_end_time:
                self.in_post_dodge_grace = False
                logger.info("[GRACE] Post-dodge grace period ended")
                return False
            return True
        return False

    def blind_forward(self):
        """Move forward despite having no optical-flow features.

        Returns
        -------
        NavigationState
            ``NavigationState.BLIND_FORWARD``.
        """
        logger.warning(
            "\u26A0\uFE0F No features — continuing blind forward motion")
        state = self.client.getMultirotorState()
        z = state.kinematics_estimated.position.z_val  # NED: z is negative up
        self.client.moveByVelocityZAsync(2,0,z,duration=2,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(False, 0),)
        self.last_movement_time = time.time()
        if not self.grace_used:
            self.just_resumed = True
            self.resume_grace_end_time = time.time() + 1.0
            self.grace_used = True
        return NavigationState.BLIND_FORWARD

    def nudge_forward(self):
        """Gently push the drone forward when stalled.

        Returns
        -------
        NavigationState
            ``NavigationState.NUDGE``.
        """
        logger.warning(
            "\u26A0\uFE0F Low flow + zero velocity — nudging forward"
        )
        state = self.client.getMultirotorState()
        z = state.kinematics_estimated.position.z_val  # NED: z is negative up
        self.client.moveByVelocityZAsync(0.5, 0, z, 1)
        self.last_movement_time = time.time()
        return NavigationState.NUDGE

    def reinforce(self):
        """Reissue the forward command to reinforce motion.

        Returns
        -------
        NavigationState
            ``NavigationState.RESUME_REINFORCE``.
        """
        logger.info("\U0001F501 Reinforcing forward motion")
        state = self.client.getMultirotorState()
        z = state.kinematics_estimated.position.z_val  # NED: z is negative up
        self.client.moveByVelocityZAsync(
            2, 0, z,
            duration=3,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(False, 0),
        )
        self.braked = False
        self.dodging = False
        self.last_movement_time = time.time()

        if not self.grace_used:
            self.just_resumed = True
            self.resume_grace_end_time = time.time() + 1.0
            self.grace_used = True
            logger.info("\U0001F552 Grace period started (first movement only)")

        return NavigationState.RESUME_REINFORCE

    def timeout_recover(self):
        """Move slowly forward after a command timeout.

        Returns
        -------
        NavigationState
            ``NavigationState.TIMEOUT_NUDGE``.
        """
        logger.warning("\u23F3 Timeout — forcing recovery motion")
        self.client.moveByVelocityAsync(0.5, 0, 0, 1)
        self.last_movement_time = time.time()
        return NavigationState.TIMEOUT_NUDGE
    

    def slam_to_goal(self, pose, goal, max_speed=0.75, threshold=0.5,
                     settle_time=1.0, velocity_threshold=0.1):
        """Move toward ``goal`` using the provided SLAM ``pose``.

        Parameters
        ----------
        pose : tuple or list
            Current SLAM pose as ``(x, y, z)`` or a 3x4 matrix.
        goal : Tuple[float, float, float]
            Target ``(x, y, z)`` position in AirSim coordinates.
        max_speed : float, optional
            Maximum forward speed in m/s.
        threshold : float, optional
            Distance threshold for considering the goal reached.
        settle_time : float, optional
            How long to wait for the drone to settle once at the goal.
        velocity_threshold : float, optional
            Minimum velocity considered as settled.

        Returns
        -------
        str
            Action string describing the command issued.
        """

        state = self.client.getMultirotorState()

        yaw = 0.0
        if pose is None:
            pos = state.kinematics_estimated.position
            x, y, z = pos.x_val, pos.y_val, pos.z_val
        else:
            if (
                isinstance(pose, (list, tuple))
                and len(pose) == 3
                and not isinstance(pose[0], (list, tuple))
            ):
                x, y, z = pose
            else:
                # Assume pose is a 3x4 matrix-like object
                x = pose[0][3]
                y = pose[1][3]
                z = pose[2][3]
                # Extract yaw from rotation matrix and apply offset
                try:
                    yaw = math.degrees(math.atan2(pose[1][0], pose[0][0]))
                except Exception:
                    yaw = 0.0
                yaw += getattr(config, "SLAM_YAW_OFFSET", 0.0)

        gx, gy, gz = goal
        dx = gx - x
        dy = gy - y
        dist = math.sqrt(dx**2 + dy**2)
        if dist < threshold:
            # Stop the drone
            self.client.moveByVelocityAsync(0, 0, 0, 0.5, vehicle_name="UAV")
            # Wait until drone settles (position within threshold and velocity near zero)
            settle_start = time.time()
            while True:
                state = self.client.getMultirotorState()
                pos = state.kinematics_estimated.position
                vel = state.kinematics_estimated.linear_velocity
                x, y = pos.x_val, pos.y_val
                vx, vy, vz = vel.x_val, vel.y_val, vel.z_val
                dist = math.sqrt((gx - x)**2 + (gy - y)**2)
                speed = math.sqrt(vx**2 + vy**2 + vz**2)
                if dist < threshold and speed < velocity_threshold:
                    break
                if time.time() - settle_start > settle_time:
                    break
                time.sleep(0.1)
            return "airsim_stop"
        vx, vy = dx / dist * max_speed, dy / dist * max_speed
        vz = 0.0  # Don't change altitude, just hold current Z
        self.client.moveByVelocityAsync(
            vx,
            vy,
            vz,
            duration=1,
            vehicle_name="UAV",
            yaw_mode=airsim.YawMode(False, yaw),
        )
        return f"airsim_nav vx={vx:.2f} vy={vy:.2f} vz={vz:.2f} dist={dist:.2f} yaw={yaw:.2f}"
