import math

def distance(arr1, arr2):
	len1 = len(arr1)
	len2 = len(arr2)

	#if (len1 > 2*len2) or (len2 > 2*len1):
	#	print '[warning] This comparison may be invalid'
	table = []
	for i in range(0, len1, 1):
		table.append(range(0, len2, 1))

	for i in range(0, len1, 1):
		for j in range(0, len2, 1):
			_min = 0
			if i > 0 and j > 0:
				_min = table[i-1][j]
				if _min > table[i-1][j-1]:
					_min = table[i-1][j-1]
				if _min > table[i][j-1]:
					_min = table[i][j-1]

			elif i == 0 and j > 0:
				_min = table[0][j-1]

			elif i > 0 and j == 0:
				_min = table[i-1][0]

			else:
				dis = abs(arr1[i][0]-arr2[j][0]) +  abs(arr1[i][1]-arr2[j][1]) +  abs(arr1[i][2]-arr2[j][2])# + abs(arr1[i][3]-arr2[j][3]) +  abs(arr1[i][4]-arr2[j][4]) +  abs(arr1[i][5]-arr2[j][5])
				table[i][j] =  dis
				continue

			dis = abs(arr1[i][0]-arr2[j][0]) +  abs(arr1[i][1]-arr2[j][1]) +  abs(arr1[i][2]-arr2[j][2]) #+ abs(arr1[i][3]-arr2[j][3]) +  abs(arr1[i][4]-arr2[j][4]) +  abs(arr1[i][5]-arr2[j][5])
			table[i][j] = dis+_min


	return table[-1][-1]
def constraint_distance(arr1, arr2, constraint=0.5, headFree=False, tailFree=False):

	ts1 = arr1
	ts2 = arr2
	infinit = float('inf')
	if len(ts1) < len(ts2):
		ts1 = arr2
		ts2 = arr1
	len1 = len(ts1)
	len2 = len(ts2)
	if float(len1)/len2 >= 1/constraint:
		return infinit
	#  ---------ts2 (ts1 must be longer than ts2)
	# | \  len2 / 
	# |      i
	# |
	# |  
	# |  len1
	# |   j
	# |
	# |
	# | /
	# ts1
	
	warpingHeight = len1-len2*constraint
	warpingWidth = len2-len1*constraint

	dt = [[infinit for j in xrange(len1)] for i in xrange(len2)]
	if headFree and tailFree:
		print "fk;lfekwf[pwkf[pew"
		for i in xrange(len2):
			for j in xrange(len1):
				# if j>=i*constraint and j<=i*constraint+warpingHeight:
				if i == 0:
					dt[i][j] = abs(ts2[i][0]-ts1[j][0]) +  abs(ts2[i][1]-ts1[j][1]) +  abs(ts2[i][2]-ts1[j][2])
				else:
					dt[i][j] = abs(ts2[i][0]-ts1[j][0]) +  abs(ts2[i][1]-ts1[j][1]) +  abs(ts2[i][2]-ts1[j][2])+min([dt[i-1][j], dt[i-1][j-1], dt[i][j-1]])
				# else:
				# 	pass

		# for line in dt:
		# 	print line
		return min(dt[-1])

	elif headFree and not tailFree:
		for i in xrange(len2):
			for j in xrange(len1):
				if j>=i*constraint and j<=i*constraint+warpingHeight:
					if i<=j*constraint+warpingWidth:
						if i == 0:
							dt[i][j] = abs(ts2[i]-ts1[j])
						else:
							dt[i][j] = abs(ts2[i]-ts1[j])+min([dt[i-1][j], dt[i-1][j-1], dt[i][j-1]])
					else:
						pass
				else:
					pass

		# for line in dt:
		# 	print line
		return dt[-1][-1]
	elif not headFree and tailFree:
		for i in xrange(len2):
			for j in xrange(len1):
				if j>=i*constraint and j<=i*constraint+warpingHeight:
					if i>=j*constraint:
						if i == 0:
							dt[i][j] = abs(ts2[i]-ts1[j])
						else:
							dt[i][j] = abs(ts2[i]-ts1[j])+min([dt[i-1][j], dt[i-1][j-1], dt[i][j-1]])
					else:
						pass
				else:
					pass


		# for line in dt:
		# 	print line
		return min(dt[-1])
	else:
		for i in xrange(len2):
			for j in xrange(len1):
				if j>=i*constraint and j<=i*constraint+warpingHeight:
					if i>=j*constraint and i<=j*constraint+warpingWidth:
						if i == 0:
							dt[i][j] = abs(ts2[i]-ts1[j])
						else:
							dt[i][j] = abs(ts2[i]-ts1[j])+min([dt[i-1][j], dt[i-1][j-1], dt[i][j-1]])

					else:
						pass
				else:
					pass


		for line in dt:
			print line
		return dt[-1][-1]
