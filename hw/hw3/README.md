# Grasp It!
The goal of the task is to grasp a ball from the table. The exercise is divided into two parts:  
 - find the ball on a table in the image plane. The ball is in the visual field of the robot and is green.
 - grasp the ball. Use the position of ball in image plane to compote its 3D position and grasp it.

![Grasp It!](https://raw.githubusercontent.com/rustlluk/pycub/master/exercises/exercise_5/exercise_5.gif)


## Task
Implement find_the_ball() to obtain 2D center of the ball in image plane  
Implement grasp() that will:
  - obtain 3D position of the ball using the center obtained from find_the_ball()
  - Look at the ball
    - you can use get_pupil_vectors() to get the vector from left pupil to the ball if needed
    - it is enough to look just once, you do not have to follow the ball later
    - there will be no check whether it is super precise, just whether you looked
  - (Select correct hand). This is optional, but based on the position of the ball it may be impossible to reach it with 
    the default end-effector of right hand (link r_hand) and you may need to change to left hand (link l_hand)
    - there is a help function to do that   
  - grasp the ball
  - Lift it up so it is at least 5cm from the table (in any direction) and the ball is still in the hand of the robot (less than 5cm from the end-effector)

The class takes three arguments
   - client - instance of pyCub class that controls the simulation
   - fake_vision - if true, saved RGB and 3D points are used. Needed for BRUTE automatic evaluation, but you can use it as well
     - with this, getting RGB is the same, but to obtain 3D point you need to call `get_3d_point(u, v)`,
       where u,v are pixels in the image. With real vision, you also need to obtain depth and call `get_3d_point(u, v, d)`, where d is depth at pixels u,v
   - idx - index of position so correct fake vision is loaded
The two function have predefined arguments and returns:
   - find_the_ball() - takes no arguments and should return (u, v) - center of the ball in image plane
   - grasp() - takes one argument `center` (the 2D center returned by find_the_ball()) and should return 0 in case of successful grasp
You are free to add any method or variable to the class, but only find_the_ball() and grasp() will be called.

**You must not** move the robot before calling find_the_ball() as BRUTE uses FAKE_VISION and it would break if you move.

The class Grasper contains several helpful methods:
  - to get RGB and Depth images
  - to deproject the 2D point
  - to close fingers
  - to set end-effector to a different one
  - to show current coordinate axes of the end-effector
  - to get vector from left pupil to ball
  - to swap quaternion representations

**The ball is always green (0, 255, 0) and it's radius is 2.5cm.**

## Scoring
 - If the ball is 5cm above the table and in the hand of the robot (less than 5cm from the end-effector) after you return from grasp(), you passed the test
 - The evaluation system will test 3 positions of the ball. Each position will be tested 3 times. If you succeed at least 
   1 out of the 3 times per position, you will get 5 points. In total you can get 3*5=15 points for this task.  
 - if you grasp the ball in all 9 (3*3) tests, you will get 2 more points -> max 17 points

## Requirements
 - Implement the two functions (find_the_ball() and grasp()) in `hw3.py`
   - changing names of functions, arguments or the file name will result in 0 points
 - Grasp the ball and move it up at least 5cm above the table
   - it will be tested with three position of the ball: [-0.35, 0.175, -0.1], [-0.35, 0, -0.1], [-0.35, -0.175, -0.1]
   - You are given the position so you can test everything offline. However, **using information about the position anyhow will result in 0 points!**
     - e.g., you should use position computed in code. Using pre-defined joint positions for grasp pose without using the computed points is forbidden
       - you can, however, use fixed joint positions for the final pose while going up from the table etc.
 - The ball must also be still in the hand
   - this is for now checked such that the ball is less than 5cm from the end-effector for the automatic evaluation. 
     But the lab tutor will check it even visually and can reduce points.
 - You can add new variables or methods to Grasper class. 
 - Do not change any variables or methods inside pyCub
   - 0 points will be given otherwise 
 - Do not change the gravity
   - 0 points will be given otherwise

**The automatic evaluation may not be perfect. If the solution works on your machine, but not in the evaluation, please contact the lab tutor.**