import sys, getopt
import h5py

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    # print 'Input file is "', inputfile
    # print 'Output file is "', outputfile

    f = h5py.File(inputfile, 'r')

    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]
    print(list(f.keys()))

    # Get the data
    data = list(f[a_group_key])
    print(data)

    f.close()

    # Write data to HDF5
    # data_file = h5py.File(outputfile, 'w')
    # data_file.create_dataset('group_name', data=data)
    # data_file.close()

if __name__ == "__main__":
    main(sys.argv[1:])



