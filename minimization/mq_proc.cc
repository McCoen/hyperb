#include <octave-3.6.2/octave/oct.h>
#include <unistd.h>
#include <spawn.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

DEFUN_DLD(mq_proc, args, nargout, "Hello World Help String") {
    Matrix a = args(0).matrix_value();

	int     fd[2], nbytes;
        pid_t   childpid;
        char readbuffer[1024], writebuffer[1024];

        pipe(fd);
        
        if ((childpid = fork()) == -1) {
		perror("fork");
		exit(1);
	}

        if (childpid == 0) {
                // Child process closes up input side of pipe 
                //close(fd[0]);

                // Send "string" through the output side of pipe 
                nbytes = read(fd[0], readbuffer, sizeof(readbuffer));
		int someArg = atoi(readbuffer);

                //printf("Child received string: %s", readbuffer);
		printf("Child received arg: %d\n", someArg);

		srand(time(NULL));
		int ans = rand() % someArg;
		sprintf(writebuffer, "%d", ans);

		write(fd[1], writebuffer, (strlen(writebuffer) + 1));
		
                //exit(0);
        } else {
                // Parent process closes up output side of pipe 
                //close(fd[1]);

                // Read in a string from the pipe 
		int parArg = 123;
		sprintf(writebuffer, "%d", parArg);
		write(fd[1], writebuffer, (strlen(writebuffer) + 1));

		sleep(1);
		nbytes = read(fd[0], readbuffer, sizeof(readbuffer));
		printf("Parent received value: %s\n", readbuffer);
		//exit(0);
        }

	return octave_value(a);
}
