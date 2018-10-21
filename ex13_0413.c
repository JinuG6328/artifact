static char help[] = "Poisson Problem in 2d and 3d with finite elements.\n\
We solve the Poisson problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\
This example supports automatic convergence estimation\n\
and eventually adaptivity.\n\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscconvest.h>

typedef struct {
  PetscScalar mu;                /* Viscosity */
  PetscScalar k;                 /* Permeability */
  PetscScalar rho;               /* Density of Rock */
  PetscScalar phi;               /* Porosity */
  PetscScalar b;                 /* Bulk Modulus */
} PhysConstants;


typedef struct {
  /* Domain and mesh definition */
  PetscInt  dim;               /* The topological mesh dimension */
  PetscBool simplex;           /* Simplicial mesh */
  PetscInt  cells[3];          /* The initial domain division */
  PetscBool adjoint;           /* Solve the adjoint problem */
  PhysConstants consts;        /* Parameter */
} AppCtx;

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = 0.0;
  return 0;
}

static PetscErrorCode trig_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  *u = 0.0;
  //for (d = 0; d < dim; ++d) *u += PetscSinReal(2.0*PETSC_PI*x[d]);
  return 0;
}

static void f0_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f0[d] = constants[0]/constants[1]*u[d]-9.8;
}

static void f1_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  PetscInt i;
  PetscReal p;
  i = uOff[1];
  p = u[i];
  for (d = 0; d < dim; ++d) f1[d] = p;
}

static void f0_q(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0)
{
  f0 = 0;
  PetscInt d;
  PetscReal *veLx = &u_x[uOff_x[0]];
  for (d = 0; d < dim; ++d) f0 += veLx[(dim+1)*d];
}

static void f1_q(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1)
{
  f1 = 0;
}

static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = 1.0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       n = 3;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim      = 2;
  options->cells[0] = 1;
  options->cells[1] = 1;
  options->cells[2] = 1;
  options->simplex  = PETSC_TRUE;
  options->adjoint  = PETSC_FALSE;

  options->consts.mu = 1.33e-4;                 /* Viscousity */
  options->consts.k =1e-16;                     /* Permeability */
  options->consts.rho = 2700;                   /* Density of Rock */
  options->consts.phi = 0.1;                   /* Porosity */
  options->consts.b = 1e-10;                    /* Bulk modulus*/

  ierr = PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex13.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-cells", "The initial mesh division", "ex13.c", options->cells, &n, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Simplicial (true) or tensor (false) mesh", "ex13.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-mu", "viscosity", "ex13.c", options->consts.mu, &options->consts.mu, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-k", "permeability", "ex13.c", options->consts.k, &options->consts.k, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-rho", "density", "ex13.c", options->consts.rho, &options->consts.rho, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-phi", "porosity", "ex13.c", options->consts.phi, &options->consts.phi, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-b", "bulk modulus", "ex13.c", options->consts.b, &options->consts.b, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* Create box mesh */
  ierr = DMPlexCreateBoxMesh(comm, user->dim, user->simplex, user->cells, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  /* Distribute mesh over processes */
  {
    DM               dmDist = NULL;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    ierr = DMPlexDistribute(*dm, 0, NULL, &dmDist);CHKERRQ(ierr);
    if (dmDist) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = dmDist;
    }
  }
  /* TODO: This should be pulled into the library */
  {
    char      convType[256];
    PetscBool flg;

    ierr = PetscOptionsBegin(comm, "", "Mesh conversion options", "DMPLEX");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-dm_plex_convert_type","Convert DMPlex to another format","ex12",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();
    if (flg) {
      DM dmConv;

      ierr = DMConvert(*dm,convType,&dmConv);CHKERRQ(ierr);
      if (dmConv) {
        ierr = DMDestroy(dm);CHKERRQ(ierr);
        *dm  = dmConv;
      }
    }
  }
  /* TODO: This should be pulled into the library */
  ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMSetApplicationContext(*dm, user);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  /* TODO: Add a hierachical viewer */
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupPrimalProblem1(PetscDS prob, AppCtx *user)
{
  const PetscInt id = 1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscDSSetResidual(prob, 0, f0_v, f1_v);CHKERRQ(ierr);
  //ierr = PetscDSSetResidual(prob, 1, f0_q, f1_q);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
 // ierr = PetscDSSetJacobian(prob, 1, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "wall", "marker", 0, 0, NULL, (void (*)(void)) trig_u, 1, &id, user);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(prob, 0, trig_u);CHKERRQ(ierr);
  ierr = PetscDSSetConstants(prob, 5, (PetscScalar *) &(user->consts));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode SetupPrimalProblem(PetscDS prob, AppCtx *user)
{
  const PetscInt id = 1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscDSSetResidual(prob, 0, f0_v, f1_v);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "wall", "marker", 0, 0, NULL, (void (*)(void)) trig_u, 1, &id, user);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(prob, 0, trig_u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, const char name[], PetscErrorCode (*setup)(PetscDS, AppCtx *), AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  PetscDS        prob;
  char           prefix[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* Create finite element */
  ierr = PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, user->dim, 1, user->simplex, name ? prefix : NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, name);CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
  ierr = (*setup)(prob, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMSetDS(cdm, prob);CHKERRQ(ierr);
    /* TODO: Check whether the boundary of coarse meshes is marked */
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;   /* Problem specification */
  SNES           snes; /* Nonlinear solver */
  Vec            u;    /* Solutions */
  //Vec            p;
  AppCtx         user; /* User-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  /* Primal system */
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &`ser, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, "potential", SetupPrimalProblem, &user);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  //ierr = DMCreateGlobalVector(dm, &p);CHKERRQ(ierr);
  ierr = VecSet(u, 0.0);CHKERRQ(ierr);
  //ierr = VecSet(p, 0.0);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "potential");CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dm, &user, &user, &user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
  ierr = SNESGetSolution(snes, &u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-potential_view");CHKERRQ(ierr);
  
  /* Cleanup */
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 2d_p1_0
    requires: triangle
    args: -potential_petscspace_order 1 -dm_refine 2 -num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_p2_0
    requires: triangle
    args: -potential_petscspace_order 2 -dm_refine 2 -num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_p3_0
    requires: triangle
    args: -potential_petscspace_order 3 -dm_refine 2 -num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q1_0
    args: -simplex 0 -potential_petscspace_order 1 -dm_refine 2 -num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q2_0
    args: -simplex 0 -potential_petscspace_order 2 -dm_refine 2 -num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q3_0
    args: -simplex 0 -potential_petscspace_order 3 -dm_refine 2 -num_refine 3 -snes_convergence_estimate
  test:
    suffix: 3d_p1_0
    requires: ctetgen
    args: -dim 3 -potential_petscspace_order 1 -num_refine 3 -snes_convergence_estimate
  test:
    suffix: 3d_p2_0
    requires: ctetgen
    args: -dim 3 -potential_petscspace_order 2 -num_refine 3 -snes_convergence_estimate
  test:
    suffix: 3d_p3_0
    requires: ctetgen
    args: -dim 3 -potential_petscspace_order 3 -dm_refine 1 -num_refine 3 -snes_convergence_estimate
  test:
    suffix: 3d_q1_0
    args: -dim 3 -simplex 0 -potential_petscspace_order 1 -dm_refine 1 -num_refine 3 -snes_convergence_estimate
  test:
    suffix: 3d_q2_0
    args: -dim 3 -simplex 0 -potential_petscspace_order 2 -dm_refine 2 -num_refine 2 -snes_convergence_estimate
  test:
    suffix: 3d_q3_0
    args: -dim 3 -simplex 0 -potential_petscspace_order 3 -num_refine 2 -snes_convergence_estimate
  test:
    suffix: 2d_p1_adj_0
    requires: triangle
    args: -potential_petscspace_order 1 -dm_refine 2 -adjoint -adjoint_petscspace_order 1 -error_petscspace_order 0

TEST*/
