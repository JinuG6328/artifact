
static char help[] = "Builds a parallel vector with 1 component on the first processor, 2 on the second, etc.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       i,N, jStart, jEnd;
  PetscScalar    one = 1.0, v;
  Vec            x;

  // Initialize PETSc
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  // Get the rank number
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  // Make Vector in X
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);

  // Set size
  ierr = VecSetSizes(x,rank+1,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);


  ierr = VecGetOwnershipRange(x,&jStart,&jEnd);CHKERRQ(ierr);

  // Set the vector elements.
  for (i=jStart; i<jEnd; i++) {
    v = (PetscScalar)(i+1);
    ierr = VecSetValues(x,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  }

  // After setting vector elements, assemby
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  /*
      View the vector; then destroy it.
  */
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);

  // Finalize
  ierr = PetscFinalize();
  return ierr;
}

