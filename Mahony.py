import math
import struct
import numpy as np
class MahonyClass:

	def __init__(self):
		self.Q = [None]*4
		self.RAD2DEG = 57.2957795
		
		self.Q[0] = 1.0
		self.Q[1] = 0.0
		self.Q[2] = 0.0
		self.Q[3] = 0.0
		self.q0q0=1.0
		self.q0q1=0.0
		self.q0q2=0.0
		self.q0q3=0.0
		self.q1q1=0.0
		self.q1q2=0.0
		self.q1q3=0.0
		self.q2q2 = 0.0
		self.q2q3 = 0.0
		self.q3q3 = 0.0

		self.twoKp = 2.0 * 1.75
		self.twoKi = 2.0 * 0.1
		self.integralFBx = 0.0
		self.integralFBy = 0.0
		self.integralFBz = 0.0
		self.count = 0
		self.beta = 0.1

		self.normMagx = 0.0
		self.normMagy = 0.0
		self.normMagz = 0.0

	def struct_isqrt(self,number):
		
		threehalfs = 1.5
		x2 = number * 0.5
		y = number
		packed_y = struct.pack('f', y) 

		i = struct.unpack('i', packed_y)[0]  # treat float's bytes as int 

		i = 0x5f3759df - (i >> 1)            # arithmetic with magic number
		packed_i = struct.pack('i', i)
		y = struct.unpack('f', packed_i)[0]  # treat int's bytes as float
		y = y * (threehalfs - (x2 * y * y))  # Newton's method
		return y

	def MahonyIMU6Dof(self, gx, gy, gz, ax, ay, az, SamplePeriod):		
		halfex = halfey = halfez = 0

		if ax != 0.0 and ay != 0.0 and az != 0.0:
			#Normalise accelerometer measurement
			recipNorm = 1/math.sqrt(ax * ax + ay * ay + az * az)
			ax *= recipNorm
			ay *= recipNorm
			az *= recipNorm


			self.q0q0 = self.Q[0] * self.Q[0]
			self.q0q1 = self.Q[0] * self.Q[1]
			self.q0q2 = self.Q[0] * self.Q[2]
			self.q0q3 = self.Q[0] * self.Q[3]
			self.q1q1 = self.Q[1] * self.Q[1]
			self.q1q2 = self.Q[1] * self.Q[2]
			self.q1q3 = self.Q[1] * self.Q[3]
			self.q2q2 = self.Q[2] * self.Q[2]
			self.q2q3 = self.Q[2] * self.Q[3]
			self.q3q3 = self.Q[3] * self.Q[3]
                        
			
			#Estimated direction of gravity
			halfvx = self.q1q3 - self.q0q2
			halfvy = self.q0q1 + self.q2q3
			#halfvz = q0q0 - 0.5f + q3q3
			halfvz = 0.5 * (self.q0q0 - self.q1q1 - self.q2q2 + self.q3q3)
			
			#Error is sum of cross product between estimated direction and measured direction of field vectors
			halfex += (ay * halfvz - az * halfvy)
			halfey += (az * halfvx - ax * halfvz)
			halfez += (ax * halfvy - ay * halfvx)
			
		if halfex != 0.0 and halfey != 0.0 and halfez != 0.0:
			#Compute and apply integral feedback if enabled
			if self.twoKi > 0.0 :
				self.integralFBx += self.twoKi * halfex * SamplePeriod # integral error scaled by Ki
				self.integralFBy += self.twoKi * halfey * SamplePeriod
				self.integralFBz += self.twoKi * halfez * SamplePeriod
				gx += self.integralFBx #apply integral feedback
				gy += self.integralFBy
				gz += self.integralFBz
			else:
				self.integralFBx = 0.0 # prevent integral windup
				self.integralFBy = 0.0
				self.integralFBz = 0.0

			#Apply proportional feedback
			gx += self.twoKp * halfex
			gy += self.twoKp * halfey
			gz += self.twoKp * halfez
			
		#Integrate rate of change of quaternion
		gx *= (0.5 * SamplePeriod)   #pre-multiply common factors
		gy *= (0.5 * SamplePeriod)
		gz *= (0.5 * SamplePeriod)

		qa = self.Q[0];
		qb = self.Q[1];
		qc = self.Q[2];
		self.Q[0] += (-qb * gx - qc * gy - self.Q[3] * gz);
		self.Q[1] += (qa * gx + qc * gz - self.Q[3] * gy);
		self.Q[2] += (qa * gy - qb * gz + self.Q[3] * gx);
		self.Q[3] += (qa * gz + qb * gy - qc * gx);
        
		#Normalise quaternion
		recipNorm = 1/math.sqrt(self.Q[0] * self.Q[0] + self.Q[1] * self.Q[1] + self.Q[2] * self.Q[2] + self.Q[3] * self.Q[3]);
		self.Q[0] *= recipNorm;
		self.Q[1] *= recipNorm;
		self.Q[2] *= recipNorm;
		self.Q[3] *= recipNorm;
		

		self.q0q0 = self.Q[0] * self.Q[0]
		self.q0q1 = self.Q[0] * self.Q[1]
		self.q0q2 = self.Q[0] * self.Q[2]
		self.q0q3 = self.Q[0] * self.Q[3]
		self.q1q1 = self.Q[1] * self.Q[1]
		self.q1q2 = self.Q[1] * self.Q[2]
		self.q1q3 = self.Q[1] * self.Q[3]
		self.q2q2 = self.Q[2] * self.Q[2]
		self.q2q3 = self.Q[2] * self.Q[3]
		self.q3q3 = self.Q[3] * self.Q[3]
	def MahonyAHRSupdate(self, gx, gy, gz, ax, ay, az, mx, my, mz, SamplePeriod):
		#float recipNorm;
		#float hx, hy, bx, bz;
		#float halfvx, halfvy, halfvz, halfwx, halfwy, halfwz;
		#float halfex, halfey, halfez;
		#float qa, qb, qc;
		    
		# Use IMU algorithm if magnetometer measurement invalid (avoids NaN in magnetometer normalisation)
		if((mx == 0.0) and  (my == 0.0) and (mz == 0.0)) :
			self.MahonyIMU6Dof(gx, gy, gz, ax, ay, az,SamplePeriod);
			return
		
		# Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)
		if not ((ax == 0.0) and (ay == 0.0) and (az == 0.0)) :

			# Normalise accelerometer measurement
			recipNorm = 1/math.sqrt(ax * ax + ay * ay + az * az)
			ax *= recipNorm
			ay *= recipNorm
			az *= recipNorm     
			# print ("Normalise ax:",ax)
			# print ("Normalise ay:",ay)
			# print ("Normalise az:",az)
			# Normalise magnetometer measurement
			recipNorm = 1/math.sqrt(mx * mx + my * my + mz * mz) 
			mx *= recipNorm
			my *= recipNorm
			mz *= recipNorm   
			# print ("Normalise mx:",mx)
			# print ("Normalise my:",my)
			# print ("Normalise mz:",mz)
                        #print("q0q0",self.q0q0)
			# Auxiliary variables to avoid repeated arithmetic
			self.q0q0 = self.Q[0] * self.Q[0]
			self.q0q1 = self.Q[0] * self.Q[1]
			self.q0q2 = self.Q[0] * self.Q[2]
			self.q0q3 = self.Q[0] * self.Q[3]
			self.q1q1 = self.Q[1] * self.Q[1]
			self.q1q2 = self.Q[1] * self.Q[2]
			self.q1q3 = self.Q[1] * self.Q[3]
			self.q2q2 = self.Q[2] * self.Q[2]
			self.q2q3 = self.Q[2] * self.Q[3]
			self.q3q3 = self.Q[3] * self.Q[3]
                        


			# Reference direction of Earth's magnetic field
			hx = 2.0 * (mx * (0.5 - self.q2q2 - self.q3q3) + my * (self.q1q2 - self.q0q3) + mz * (self.q1q3 + self.q0q2))
			hy = 2.0 * (mx * (self.q1q2 + self.q0q3) + my * (0.5 - self.q1q1 - self.q3q3) + mz * (self.q2q3 - self.q0q1))
			bx = math.sqrt(hx * hx + hy * hy)
			bz = 2.0 * (mx * (self.q1q3 - self.q0q2) + my * (self.q2q3 + self.q0q1) + mz * (0.5 - self.q1q1 - self.q2q2))
			

			# Estimated direction of gravity and magnetic field
			halfvx = self.q1q3 - self.q0q2
			halfvy = self.q0q1 + self.q2q3
			halfvz = self.q0q0 - 0.5 + self.q3q3
			halfwx = bx * (0.5 - self.q2q2 - self.q3q3) + bz * (self.q1q3 - self.q0q2)
			halfwy = bx * (self.q1q2 - self.q0q3) + bz * (self.q0q1 + self.q2q3)
			halfwz = bx * (self.q0q2 + self.q1q3) + bz * (0.5 - self.q1q1 - self.q2q2)



			# Error is sum of cross product between estimated direction and measured direction of field vectors
			halfex = (ay * halfvz - az * halfvy) + (my * halfwz - mz * halfwy)
			halfey = (az * halfvx - ax * halfvz) + (mz * halfwx - mx * halfwz)
			halfez = (ax * halfvy - ay * halfvx) + (mx * halfwy - my * halfwx)

			# Compute and apply integral feedback if enabled
			if(self.twoKi > 0.0):
				self.integralFBx = self.integralFBx + (self.twoKi * halfex *  SamplePeriod)	# integral error scaled by Ki
				self.integralFBy = self.integralFBx + (self.twoKi * halfey *  SamplePeriod)  #1.0 delete
				self.integralFBz = self.integralFBx + (self.twoKi * halfez * SamplePeriod)
				gx += self.integralFBx	# apply integral feedback
				gy += self.integralFBy
				gz += self.integralFBz

			
			else:
				self.integralFBx = 0.0	# prevent integral windup
				self.integralFBy = 0.0
				self.integralFBz = 0.0

			# Apply proportional feedback
			gx += self.twoKp * halfex
			gy += self.twoKp * halfey
			gz += self.twoKp * halfez
		

		
		# Integrate rate of change of quaternion
		gx *= (0.5 * SamplePeriod)		# pre-multiply common factors
		gy *= (0.5 * SamplePeriod)   #1.0 delete
		gz *= (0.5 * SamplePeriod)
		qa = self.Q[0]
		qb = self.Q[1]
		qc = self.Q[2]

		self.Q[0] += (-qb * gx - qc * gy - self.Q[3] * gz)
		self.Q[1] += (qa * gx + qc * gz - self.Q[3] * gy)
		self.Q[2] += (qa * gy - qb * gz + self.Q[3] * gx)
		self.Q[3] += (qa * gz + qb * gy - qc * gx)
		
		# Normalise quaternion
		recipNorm = 1/math.sqrt( self.Q[0] * self.Q[0] + self.Q[1] * self.Q[1] + self.Q[2] * self.Q[2] + self.Q[3] * self.Q[3])
		self.Q[0] *= recipNorm
		self.Q[1] *= recipNorm
		self.Q[2] *= recipNorm
		self.Q[3] *= recipNorm

		# self.q0q0 = self.Q[0] * self.Q[0]
		# self.q0q1 = self.Q[0] * self.Q[1]
		# self.q0q2 = self.Q[0] * self.Q[2]
		# self.q0q3 = self.Q[0] * self.Q[3]
		# self.q1q1 = self.Q[1] * self.Q[1]
		# self.q1q2 = self.Q[1] * self.Q[2]
		# self.q1q3 = self.Q[1] * self.Q[3]
		# self.q2q2 = self.Q[2] * self.Q[2]
		# self.q2q3 = self.Q[2] * self.Q[3]
		# self.q3q3 = self.Q[3] * self.Q[3]
	def MadgwickAHRSupdate(self, gx, gy, gz, ax, ay, az, mx, my, mz, SamplePeriod):


		# Rate of change of Quaternion from gyroscope

		QDot1 = 0.5 * (-self.Q[1] * gx - self.Q[2] * gy - self.Q[3] * gz)
		QDot2 = 0.5 * (self.Q[0] * gx + self.Q[2] * gz - self.Q[3] * gy)
		QDot3 = 0.5 * (self.Q[0] * gy - self.Q[1] * gz + self.Q[3] * gx)
		QDot4 = 0.5 * (self.Q[0] * gz + self.Q[1] * gy - self.Q[2] * gx)

		if not ((ax == 0.0) and (ay == 0.0) and (az == 0.0)):
			recipNorm = self.struct_isqrt(ax * ax + ay * ay + az * az)

			ax *= recipNorm
			ay *= recipNorm
			az *= recipNorm

			recipNorm = self.struct_isqrt(mx * mx + my * my + mz * mz)
			mx *= recipNorm
			my *= recipNorm
			mz *= recipNorm
			self.normMagx = mx
			self.normMagy = my
			self.normMagz = mz

			_2q0mx = 2.0 * self.Q[0] * mx
			_2q0my = 2.0 * self.Q[0] * my
			_2q0mz = 2.0 * self.Q[0] * mz
			_2q1mx = 2.0 * self.Q[1] * mx
			_2q0 = 2.0 * self.Q[0]
			_2q1 = 2.0 * self.Q[1]
			_2q2 = 2.0 * self.Q[2]
			_2q3 = 2.0 * self.Q[3];
			_2q0q2 = 2.0 * self.Q[0] * self.Q[2]
			_2q2q3 = 2.0 * self.Q[2] * self.Q[3]
			self.q0q0 = self.Q[0] * self.Q[0]
			self.q0q1 = self.Q[0] * self.Q[1]
			self.q0q2 = self.Q[0] * self.Q[2]
			self.q0q3 = self.Q[0] * self.Q[3]
			self.q1q1 = self.Q[1] * self.Q[1]
			self.q1q2 = self.Q[1] * self.Q[2]
			self.q1q3 = self.Q[1] * self.Q[3]
			self.q2q2 = self.Q[2] * self.Q[2]
			self.q2q3 = self.Q[2] * self.Q[3]
			self.q3q3 = self.Q[3] * self.Q[3]

			hx = mx * self.q0q0 - _2q0my * self.Q[3] + _2q0mz * self.Q[2] + mx * self.q1q1 + _2q1 * my * self.Q[2] + _2q1 * mz * self.Q[3] - mx * self.q2q2 - mx * self.q3q3
			hy = _2q0mx * self.Q[3] + my * self.q0q0 - _2q0mz * self.Q[1] + _2q1mx * self.Q[2] - my * self.q1q1 + my * self.q2q2 + _2q2 * mz * self.Q[3] - my * self.q3q3
			_2bx = math.sqrt(hx * hx + hy * hy)
			_2bz = -_2q0mx * self.Q[2] + _2q0my * self.Q[1] + mz * self.q0q0 + _2q1mx * self.Q[3] - mz * self.q1q1 + _2q2 * my * self.Q[3] - mz * self.q2q2 + mz * self.q3q3
			_4bx = 2.0 * _2bx
			_4bz = 2.0 * _2bz


			# Gradient decent algorithm corrective step
			s0 = -_2q2 * (2.0 * self.q1q3 - _2q0q2 - ax) + _2q1 * (2.0 * self.q0q1 + _2q2q3 - ay) - _2bz * self.Q[2] * (_2bx * (0.5 - self.q2q2 - self.q3q3) + _2bz * (self.q1q3 - self.q0q2) - mx) + (-_2bx * self.Q[3] + _2bz * self.Q[1]) * (_2bx * (self.q1q2 - self.q0q3) + _2bz * (self.q0q1 + self.q2q3) - my) + _2bx * self.Q[2] * (_2bx * (self.q0q2 + self.q1q3) + _2bz * (0.5 - self.q1q1 - self.q2q2) - mz)
			s1 = _2q3 * (2.0 * self.q1q3 - _2q0q2 - ax) + _2q0 * (2.0 * self.q0q1 + _2q2q3 - ay) - 4.0 * self.Q[1] * (1 - 2.0 *self.q1q1 - 2.0 * self.q2q2 - az) + _2bz * self.Q[3] * (_2bx * (0.5 - self.q2q2 - self.q3q3) + _2bz * (self.q1q3 - self.q0q2) - mx) + (_2bx * self.Q[2] + _2bz * self.Q[0]) * (_2bx * (self.q1q2 - self.q0q3) + _2bz * (self.q0q1 + self.q2q3) - my) + (_2bx * self.Q[3] - _4bz * self.Q[1]) * (_2bx * (self.q0q2 + self.q1q3) + _2bz * (0.5 - self.q1q1 - self.q2q2) - mz)
			s2 = -_2q0 * (2.0 * self.q1q3 - _2q0q2 - ax) + _2q3 * (2.0 * self.q0q1 + _2q2q3 - ay) - 4.0 * self.Q[2] * (1 - 2.0 * self.q1q1 - 2.0 * self.q2q2 - az) + (-_4bx * self.Q[2] - _2bz * self.Q[0]) * (_2bx * (0.5 - self.q2q2 - self.q3q3) + _2bz * (self.q1q3 - self.q0q2) - mx) + (_2bx * self.Q[1] + _2bz * self.Q[3]) * (_2bx * (self.q1q2 - self.q0q3) + _2bz * (self.q0q1 + self.q2q3) - my) + (_2bx * self.Q[0] - _4bz * self.Q[2]) * (_2bx * (self.q0q2 + self.q1q3) + _2bz * (0.5 - self.q1q1 - self.q2q2) - mz)
			s3 = _2q1 * (2.0 * self.q1q3 - _2q0q2 - ax) + _2q2 * (2.0 * self.q0q1 + _2q2q3 - ay) + (-_4bx * self.Q[3] + _2bz * self.Q[1]) * (_2bx * (0.5 - self.q2q2 - self.q3q3) + _2bz * (self.q1q3 - self.q0q2) - mx) + (-_2bx * self.Q[0] + _2bz * self.Q[2]) * (_2bx * (self.q1q2 - self.q0q3) + _2bz * (self.q0q1 + self.q2q3) - my) + _2bx * self.Q[1] * (_2bx * (self.q0q2 + self.q1q3) + _2bz * (0.5 - self.q1q1 - self.q2q2) - mz)
			recipNorm = self.struct_isqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3) # normalise step magnitude
			s0 *= recipNorm
			s1 *= recipNorm
			s2 *= recipNorm
			s3 *= recipNorm

			# Apply feedback step
			QDot1 -= self.beta * s0
			QDot2 -= self.beta * s1
			QDot3 -= self.beta * s2
			QDot4 -= self.beta * s3
			# Integrate rate of change of Quaternion to yield Quaternion
		self.Q[0] += QDot1 * (1.0 * SamplePeriod)
		self.Q[1] += QDot2 * (1.0 * SamplePeriod)
		self.Q[2] += QDot3 * (1.0 * SamplePeriod)
		self.Q[3] += QDot4 * (1.0 * SamplePeriod)

		# Normalise Quaternion
		recipNorm = self.struct_isqrt(self.Q[0] * self.Q[0] + self.Q[1] * self.Q[1] + self.Q[2] * self.Q[2] + self.Q[3] * self.Q[3])

		self.Q[0] *= recipNorm
		self.Q[1] *= recipNorm
		self.Q[2] *= recipNorm
		self.Q[3] *= recipNorm
		# print self.Q[0],self.Q[1],self.Q[2],self.Q[3]
	def quatern2ReverseRotMat(self):

		rotmat = [[0,0,0],[0,0,0],[0,0,0]]
		
		rotmat[0][0] = 2*( self.q0q0 + self.q1q1 ) - 1
		rotmat[0][1] = 2*( self.q1q2 - self.q0q3 )
		rotmat[0][2] = 2*( self.q1q3 + self.q0q2 )
		rotmat[1][0] = 2*( self.q1q2 + self.q0q3)
		rotmat[1][1] = 2*( self.q0q0 + self.q2q2 ) - 1
		rotmat[1][2] = 2*( self.q2q3 - self.q0q1 )
		rotmat[2][0] = 2*( self.q1q3 - self.q0q2 )
		rotmat[2][1] = 2*( self.q2q3 + self.q0q1)
		rotmat[2][2] = 2*( self.q0q0 + self.q3q3 ) - 1

		return rotmat

	def quatern2euler(self):
		angles = [None] * 3
                
		angles[0] = math.atan2(2 * self.q1q2 + 2 * self.q0q3, -2 * self.q2q2 - 2 * self.q3q3 + 1) # yaw
		angles[1] = -1*math.asin(2 * self.q1q3 - 2 * self.q0q2) # pitch
		angles[2] = math.atan2(2 * self.q2q3 + 2 * self.q0q1, -2 * self.q1q1 - 2 * self.q2q2 + 1)

		angles[0] *= self.RAD2DEG
		angles[1] *= self.RAD2DEG
		angles[2] *= self.RAD2DEG

		# angles[0] = angles[0]
		# angles[1] = angles[1]
		# angles[2] = angles[2]
		
		'''
		self.count = self.count + 1
		print("NUM:",self.count)
		'''
		#print("ANGLE",angles)
		return angles

	def eular2quatern(self,angles):

		t0 = math.cos(angles[0]*0.5/self.RAD2DEG)
		t1 = math.sin(-1*angles[0]*0.5/self.RAD2DEG)		
		t2 = math.cos(angles[2]*0.5/self.RAD2DEG)#1
		t3 = math.sin(angles[2]*0.5/self.RAD2DEG)#0		
		t4 = math.cos(angles[1]*0.5/self.RAD2DEG)#1
		t5 = math.sin(angles[1]*0.5/self.RAD2DEG)#0
		

#		Qtmp = self.Q

		self.Q[0] = t0 * t2 * t4 + t1 * t3 * t5 # t0*1*1 + t1*0*0 = t1
		self.Q[1] = t0 * t3 * t4 - t1 * t2 * t5 # t0*0*1 - t1*t2*0 = 0
		self.Q[2] = t0 * t2 * t5 + t1 * t3 * t4 # t0*1*0 + t1*0*1 = 0
		self.Q[3] = (t1 * t2 * t4 - t0 * t3 * t5) # t1*1*1 - t0*0*0 = t1


		recipNorm = 1/math.sqrt(self.Q[0] * self.Q[0] + self.Q[1] * self.Q[1] + self.Q[2] * self.Q[2] + self.Q[3] * self.Q[3]);
		self.Q[0] *= recipNorm;
		self.Q[1] *= recipNorm;
		self.Q[2] *= recipNorm;
		self.Q[3] *= recipNorm

		self.q0q0 = self.Q[0] * self.Q[0]
		self.q0q1 = self.Q[0] * self.Q[1]
		self.q0q2 = self.Q[0] * self.Q[2]
		self.q0q3 = self.Q[0] * self.Q[3]
		self.q1q1 = self.Q[1] * self.Q[1]
		self.q1q2 = self.Q[1] * self.Q[2]
		self.q1q3 = self.Q[1] * self.Q[3]
		self.q2q2 = self.Q[2] * self.Q[2]
		self.q2q3 = self.Q[2] * self.Q[3]
		self.q3q3 = self.Q[3] * self.Q[3]
		
#		print (Qtmp,self.Q)
		
	def getnormMagValue(self):
		return [self.normMagx,self.normMagy,self.normMagz]

	def InverseQuatern(self):
		rotmat = [[0,0,0],[0,0,0],[0,0,0]]
		
		rotmat[0][0] = 2*( self.q0q0 + self.q1q1 ) - 1
		rotmat[0][1] = 2*( self.q1q2 + self.q0q3)
		rotmat[0][2] = 2*( self.q1q3 - self.q0q2 )
		rotmat[1][0] = 2*( self.q1q2 - self.q0q3 )
		rotmat[1][1] = 2*( self.q0q0 + self.q2q2 ) - 1
		rotmat[1][2] = 2*( self.q2q3 + self.q0q1)
		rotmat[2][0] = 2*( self.q1q3 + self.q0q2 )		
		rotmat[2][1] = 2*( self.q2q3 - self.q0q1 )		
		rotmat[2][2] = 2*( self.q0q0 + self.q3q3 ) - 1

		return np.asmatrix(rotmat)

	def EularRotateMatrixInverse(self,angles):
		rotmat = [[0,0,0],[0,0,0],[0,0,0]]
		Yaw =  angles[0] * 0.01745329251
		Pitch =  angles[1] * 0.01745329251
		Roll =  angles[2] * 0.01745329251

		cosYaw = math.cos(Yaw)
		cosPitch = math.cos(Pitch)
		cosRoll = math.cos(Roll)
		sinYaw = math.sin(Yaw)
		sinPitch = math.sin(Pitch)
		sinRoll = math.sin(Roll)


		rotmat[0][0] = cosYaw * cosPitch 
		rotmat[0][1] = sinYaw * cosPitch
		rotmat[0][2] = -sinPitch
		rotmat[1][0] = -sinYaw*cosRoll+cosYaw*sinPitch*sinRoll
		rotmat[1][1] = cosYaw * cosRoll + sinYaw*sinPitch*sinRoll
		rotmat[1][2] = cosPitch  * sinRoll 
		rotmat[2][0] = sinYaw * sinRoll + cosYaw*sinPitch* cosRoll	
		rotmat[2][1] = -cosYaw * sinRoll + sinYaw * sinPitch * cosRoll		
		rotmat[2][2] = cosPitch  * cosRoll

		return np.asmatrix(rotmat)