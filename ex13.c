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
  PhysConstants consts;
} AppCtx;

static PetscErrorCode trig_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  *u = 0.0;
  for (d = 0; d < dim; ++d) *u += PetscSinReal(2.0*PETSC_PI*x[d]);
  return 0;
}

static PetscErrorCode boundary_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  // PetscInt d;
  // *u = 1.0;
  // for (d = 0; d < dim; ++d) *u += PetscSinReal(2.0*PETSC_PI*x[d]);
  // return 0;

  PetscInt d;
  for (d = 0; d < dim; ++d) u[d] = 0;
  u[0] = 1.0;
  return 0;
}

static void f0_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f0[d] = u[d];

}

static void f1_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  PetscInt i = uOff[1];
  PetscReal p = u[i];
  PetscReal mu = constants[0]; /* Viscousity */
  PetscReal k = constants[1]; /* Permeability */
  // PetscReal rho = constants[2]; /* Density of Rock */
  // PetscReal phi = constants[3]; /* Porosity */
  // PetscReal b = constants[4]; /* Bulk modulus */                  
  
  for (d = 0; d < dim; ++d) f1[d] = - k/mu*p;
}

static void f0_q(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  //const PetscReal * velx = & u_x[uOff_x[0]];
  //for (d = 0, f0[0] = 0.0; d < dim; ++d) f0[0] += velx[d*dim+d];
  
  for (d = 0, f0[0] = 0.0; d < dim; ++d) f0[0] += u_x[d*dim+d];

}

static void f1_q(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  
  for (d = 0; d < dim; ++d) f1[d] = 0.0;

}

/* < u, v > */
static void g0_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  PetscInt d;
  
  for (d = 0; d < dim; ++d) g0[d*dim+d] = 1.0; // 

}

/* < \nabla\cdot v, p >
    NcompI = dim, NcompJ = 1 */
void g2_vp(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscInt d;
  //PetscReal mu = constants[0]; /* Viscousity */
  //PetscReal k = constants[1]; /* Permeability */
  
  for (d = 0; d < dim; ++d) g2[d*dim+d] = 1.0; /* \frac{\partial\psi^{u_d}}{\partial x_d} */
  
}

/* < \nabla\cdot u , q >
   NcompI = 1, NcompJ = dim */
void g1_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt d;
  
  for (d = 0; d < dim; ++d) g1[d*dim+d] = 1.0; /* \frac{\partial\phi^{u_d}}{\partial x_d} */
  
}

/* k/mu*g v*n ds */
static void f0_bd_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  PetscReal mu = constants[0]; /* Viscousity */
  PetscReal k = constants[1]; /* Permeability */
  // for (d = 0, f0[0] = 0.0; d < dim; ++d) f0[d] += n[d]*9.8*k/mu*x[2];
  for (d = 0, f0[0] = 0.0; d < dim; ++d) f0[d] = n[d]*9.8*k/mu*x[2];
}

static void f1_bd_zero(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt comp;
  for (comp = 0; comp < dim; ++comp) f1[comp] = 0.0;
}

/* k/mu*g v*n ds */
// void g1_boundary(PetscInt dim, PetscInt Nf, PetscInt NfAux,
//            const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
//            const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
//            PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
// {
//   PetscInt d;
  
//   for (d = 0; d < dim; ++d) g1[d*dim+d] = 1.0; /* \frac{\partial\phi^{u_d}}{\partial x_d} */
  
// }

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       n = 3;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim      = 3;
  options->cells[0] = 1;
  options->cells[1] = 1;
  options->cells[2] = 1;
  options->simplex  = PETSC_FALSE;
  

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
// Sepreate marker

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
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupPrimalProblem(PetscDS prob, AppCtx *user)
{
  const PetscInt id = 6;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscDSSetResidual(prob, 0, f0_v, f1_v);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 1, f0_q, f1_q);CHKERRQ(ierr);
  
  ierr = PetscDSSetJacobian(prob, 0, 0, g0_vu, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL, g2_vp, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 1, 0, NULL, g1_qu, NULL, NULL);CHKERRQ(ierr);
  
  ierr = PetscDSSetBdResidual(prob, 0, f0_bd_v, f1_bd_zero);CHKERRQ(ierr);
  
  ierr = PetscDSSetBdJacobian(prob,0, 0, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetBdJacobian(prob,0, 1, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetBdJacobian(prob,1, 0, NULL, NULL, NULL, NULL);CHKERRQ(ierr);

  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "velocity", "marker", 0, 0, NULL, (void (*)(void)) boundary_u, 1, &id, user);CHKERRQ(ierr);
  // // ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL, "pressure", "Faces", 0, 0, NULL, NULL, 0, NULL, user);CHKERRQ(ierr);
  // // DM_BC_ESSENTIAL_FIELD
  // // How can I set the specific boundary on x-driection?


  // Faces
  // ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "fixed", "Faces", 0, Ncomp, components, (void (*)(void)) zero, Nfid, fid, NULL);CHKERRQ(ierr);
  // ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL, "traction", "Faces", 0, Ncomp, components, NULL, Npid, pid, NULL);CHKERRQ(ierr);

  ierr = PetscDSSetExactSolution(prob, 0, trig_u);CHKERRQ(ierr);
  ierr = PetscDSSetConstants(prob, 5, (PetscScalar *) &(user->consts));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode SetupDiscretization(DM dm, const char name[], PetscErrorCode (*setup)(PetscDS, AppCtx *), AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe[2];  // Editing
  PetscDS        prob;
  char           prefix[PETSC_MAX_PATH_LEN];
  MPI_Comm        comm;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* Create finite element */
  ierr = PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);

  ierr = PetscFECreateDefault(comm, user->dim, user->dim, user->simplex, "vel_", PETSC_DEFAULT, &fe[0]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[0], "velocity");CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, user->dim, 1, user->simplex, "pres_", PETSC_DEFAULT, &fe[1]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[1], "pressure");CHKERRQ(ierr);
  // Add fe[2]; and FECreateDefault and ObjectSetname
  // Not including FESetQuadrature



  /* Set discretization and boundary conditions for each mesh */
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe[0]);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 1, (PetscObject) fe[1]);CHKERRQ(ierr);
  // Add Discretization

  ierr = (*setup)(prob, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMSetDS(cdm, prob);CHKERRQ(ierr);
    /* TODO: Check whether the boundary of coarse meshes is marked */
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }

  ierr = PetscFEDestroy(&fe[0]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe[1]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;   /* Problem specification */
  SNES           snes; /* Nonlinear solver */
  Vec            u;    /* Solutions */
  AppCtx         user; /* User-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  /* Primal system */
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);

  // {
  DMLabel         label;
  //   IS              is;
  ierr = DMCreateLabel(dm, "boundary");CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "boundary", &label);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(dm, 1, label);CHKERRQ(ierr);
  //   {
  //     ierr = DMGetStratumIS(dm, "boundary", 1,  &is);CHKERRQ(ierr);
  //     ierr = DMCreateLabel(dm,"Faces");CHKERRQ(ierr);
  //     if (is) {
  //       PetscInt        d, f, Nf;
  //       const PetscInt *faces;
  //       PetscInt        csize;
  //       PetscSection    cs;
  //       Vec             coordinates ;
  //       DM              cdm;
  //       ierr = ISGetLocalSize(is, &Nf);CHKERRQ(ierr);
  //       ierr = ISGetIndices(is, &faces);CHKERRQ(ierr);
  //       ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  //       ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  //       ierr = DMGetDefaultSection(cdm, &cs);CHKERRQ(ierr);
  //       /* Check for each boundary face if any component of its centroid is either 0.0 or 1.0 */
  //       for (f = 0; f < Nf; ++f) {
  //         PetscReal   faceCoord;
  //         PetscInt    b,v;
  //         PetscScalar *coords = NULL;
  //         PetscInt    Nv;
  //         ierr = DMPlexVecGetClosure(cdm, cs, coordinates, faces[f], &csize, &coords);CHKERRQ(ierr);
  //         Nv   = csize/dim; /* Calculate mean coordinate vector */
  //         for (d = 0; d < dim; ++d) {
  //           faceCoord = 0.0;
  //           for (v = 0; v < Nv; ++v) faceCoord += PetscRealPart(coords[v*dim+d]);
  //           faceCoord /= Nv;
  //           for (b = 0; b < 2; ++b) {
  //             if (PetscAbs(faceCoord - b) < PETSC_SMALL) { /* domain have not been set yet, still [0,1]^3 */
  //               ierr = DMSetLabelValue(dm, "Faces", faces[f], d*2+b+1);CHKERRQ(ierr);
  //             }
  //           }
  //         }
  //         ierr = DMPlexVecRestoreClosure(cdm, cs, coordinates, faces[f], &csize, &coords);CHKERRQ(ierr);
  //       }
  //       ierr = ISRestoreIndices(is, &faces);CHKERRQ(ierr);
  //     }
  //     ierr = ISDestroy(&is);CHKERRQ(ierr);
  //     ierr = DMGetLabel(dm, "Faces", &label);CHKERRQ(ierr);
  //     ierr = DMPlexLabelComplete(dm, label);CHKERRQ(ierr);
  //   }
  // }
  // {
  //   PetscInt    dimEmbed, i;
  //   PetscInt    nCoords;
  //   PetscScalar *coords,bounds[] = {0,1,-.5,.5,-.5,.5,}; /* x_min,x_max,y_min,y_max */
  //   Vec         coordinates;
  //   bounds[1] = Lx;
  //   if (run_type==1) {
  //     for (i = 0; i < 2*dim; i++) bounds[i] = (i%2) ? 1 : 0;
  //   }
  //   ierr = DMGetCoordinatesLocal(dm,&coordinates);CHKERRQ(ierr);
  //   ierr = DMGetCoordinateDim(dm,&dimEmbed);CHKERRQ(ierr);
  //   if (dimEmbed != dim) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"dimEmbed != dim %D",dimEmbed);
  //   ierr = VecGetLocalSize(coordinates,&nCoords);CHKERRQ(ierr);
  //   if (nCoords % dimEmbed) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Coordinate vector the wrong size");
  //   ierr = VecGetArray(coordinates,&coords);CHKERRQ(ierr);
  //   for (i = 0; i < nCoords; i += dimEmbed) {
  //     PetscInt    j;
  //     PetscScalar *coord = &coords[i];
  //     for (j = 0; j < dimEmbed; j++) {
  //       coord[j] = bounds[2 * j] + coord[j] * (bounds[2 * j + 1] - bounds[2 * j]);
  //     }
  //   }
  //   ierr = VecRestoreArray(coordinates,&coords);CHKERRQ(ierr);
  //   ierr = DMSetCoordinatesLocal(dm,coordinates);CHKERRQ(ierr);
  // }


  ierr = SetupDiscretization(dm, "potential", SetupPrimalProblem, &user);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);

  ierr = VecSet(u, 0.0);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) u, "potential");CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dm, &user, &user, &user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
  ierr = SNESGetSolution(snes, &u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-potential_view");CHKERRQ(ierr);
   // 
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
    timeoutfactor: 2
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
    suffix: 2d_p1_spectral_0
    requires: triangle fftw !complex
    args: -potential_petscspace_order 1 -dm_refine 6 -spectral -fft_view
  test:
    suffix: 2d_p1_spectral_1
    requires: triangle fftw !complex
    nsize: 2
    args: -potential_petscspace_order 1 -dm_refine 2 -spectral -fft_view
  test:
    suffix: 2d_p1_adj_0
    requires: triangle
    args: -potential_petscspace_order 1 -dm_refine 2 -adjoint -adjoint_petscspace_order 1 -error_petscspace_order 0

TEST*/
