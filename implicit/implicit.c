static char help[] = "Solves a tridiagonal linear system.\n\n";
#include<petscksp.h>
#include<petscmath.h>
#include<petscviewerhdf5.h>
#include<math.h>

#define PI      3.1415926
#define GRID    100
#define RHO     1.0
#define K       1.0
#define C       1.0
#define XEND    1.0
#define TSTEP   100000
#define TEND    2.0
#define TOL     1.e-8
int main(int argc, char **args){
    
    
/* ******************1定义变量*********************** */
    Vec uj, uj_old, f, info;
    Mat A;
    PetscErrorCode ierr;
    PetscInt i, rank, START = 0, END=GRID, pos, index, iter = 0, col[3];
    PetscBool r = PETSC_FALSE;
    PetscReal CFL, dx, dt, u0, f0, t = 0, t_start=0, t_end=TEND;
    PetscScalar coef[3], zero = 0.0;
    KSP ksp;
    PC pc;
    PetscViewer h5;
/* *********************************************** */
    
    
/* ******************2初始化参数********************* */
    ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;    //初始化Petsc
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL,NULL,"-r",&r,NULL);CHKERRQ(ierr);    //从命令行读取是否重启
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);    //设置并行MPI参数
/* *********************************************** */
    

/* *************3计算并打印初始化变量信息*************** */
    dx = 1.0/GRID;    //计算dx&dt
    dt = (t_end - t_start)/TSTEP;
    CFL= K*dt/(RHO*C*dx*dx);    //计算CFL值
    ierr = PetscPrintf(PETSC_COMM_WORLD,"dx = %f\t",dx);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"dt = %f\t",dt);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"CFL = %f\n",CFL);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "grid = %d tstep = %d\n", GRID, TSTEP);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"restart is %d\n",r);CHKERRQ(ierr);
/* *********************************************** */
    
    
/* ******************4初始化参数********************* */
    ierr = VecCreate(PETSC_COMM_WORLD,&uj);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD,&info);CHKERRQ(ierr);
    ierr = VecSetSizes(uj,PETSC_DECIDE,GRID+1);CHKERRQ(ierr);
    ierr = VecSetSizes(info, 3, PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(uj);CHKERRQ(ierr);
    ierr = VecSetFromOptions(info);CHKERRQ(ierr);
    ierr = VecDuplicate(uj,&uj_old);CHKERRQ(ierr);
    ierr = VecDuplicate(uj,&f);CHKERRQ(ierr);

    ierr = VecGetOwnershipRange(uj,&START,&END);CHKERRQ(ierr);
    ierr = VecGetLocalSize(uj,&pos);CHKERRQ(ierr);

    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatSetSizes(A,pos,pos,GRID+1,GRID+1);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);
/* *********************************************** */

    
/* ****************5初始化三对角矩阵****************** */
    if (!START){
      START = 1;
      i = 0; col[0] = 0; col[1] = 1; coef[0] = 1+2.0*CFL; coef[1] = -CFL;
      ierr   = MatSetValues(A,1,&i,2,col,coef,INSERT_VALUES);CHKERRQ(ierr);
    }
    
    if (END == GRID+1){
      END = GRID;
      i = GRID; col[0] = GRID-1; col[1] = GRID; coef[0] = -CFL; coef[1] = 1+2.0*CFL;
      ierr = MatSetValues(A,1,&i,2,col,coef,INSERT_VALUES);CHKERRQ(ierr);
    }

    coef[0] = -CFL; coef[1] = 1+2.0*CFL; coef[2] = -CFL;
    for (i=START; i<END; i++){
      col[0] = i-1; col[1] = i; col[2] = i+1;
      ierr = MatSetValues(A,1,&i,3,col,coef,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
/* *********************************************** */
    
    
/* ***********6根据-r参数对uj_old数据初始化************** */
    if(r){
        ierr = VecSet(uj_old, zero);CHKERRQ(ierr);
        if(rank == 0){
          for(int i = 1; i < END; i++){
            u0 = exp(i*dx);
              ierr = VecSetValues(uj_old, 1, &i, &u0, INSERT_VALUES);CHKERRQ(ierr);
          }
        }
        ierr = VecAssemblyBegin(uj_old);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(uj_old);CHKERRQ(ierr);
    }
    else{
        ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"implicit.h5", FILE_MODE_READ, &h5);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject) uj_old, "uj");CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject) info, "info");CHKERRQ(ierr);
        ierr = VecLoad(info, h5);CHKERRQ(ierr);
        ierr = VecLoad(uj_old, h5);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&h5);CHKERRQ(ierr);
        index=0;
        ierr = VecGetValues(info,1,&index,&dx);CHKERRQ(ierr);    /*将第一个值赋给dx*/
        index=index+1;
        ierr = VecGetValues(info,1,&index,&dt);CHKERRQ(ierr);    /*将第二个值赋给dt*/
        index=index+1;
        ierr = VecGetValues(info,1,&index,&t);CHKERRQ(ierr);    /*将第三个值赋给t*/
        index= 0;
    }
    ierr = VecSet(f, zero);CHKERRQ(ierr);
    if(rank == 0){
      for(i = 1; i < END; i++){
        f0 = dt*sin(i*dx*PI);
        ierr = VecSetValues(f, 1, &i, &f0, INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(f);CHKERRQ(ierr);
/* *********************************************** */
    
    
/* **************7创建隐式求解ksp空间***************** */
    ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,TOL,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);    /*设置各种误差值*/
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    
/* *********************************************** */
        
        
/* ****************8迭代计算uj值********************* */
    while(PetscAbsReal(t)<=TEND){
       t += dt;    /*时间向前走*/
       ierr = VecAXPY(uj_old,1.0,f);CHKERRQ(ierr);
       ierr = KSPSolve(ksp,uj_old,uj);CHKERRQ(ierr);
       ierr = VecSetValues(uj, 1, &START, &zero, INSERT_VALUES);CHKERRQ(ierr);
       ierr = VecSetValues(uj, 1, &END, &zero, INSERT_VALUES);CHKERRQ(ierr);
       ierr = VecAssemblyBegin(uj);CHKERRQ(ierr);
       ierr = VecAssemblyEnd(uj);CHKERRQ(ierr);
       ierr = VecCopy(uj,uj_old);CHKERRQ(ierr);
       iter += 1;
       if((iter % 10) == 0){
            index = 0;
            ierr = VecSetValues(info,1,&index,&dx,INSERT_VALUES);CHKERRQ(ierr);
            index += 1;
            ierr = VecSetValues(info,1,&index,&dt,INSERT_VALUES);CHKERRQ(ierr);
            index += 1;
            ierr = VecSetValues(info,1,&index,&t,INSERT_VALUES);CHKERRQ(ierr);
            index=0;
           
            ierr = VecAssemblyBegin(info);CHKERRQ(ierr);
            ierr = VecAssemblyEnd(info);CHKERRQ(ierr);

            ierr = PetscViewerCreate(PETSC_COMM_WORLD,&h5);CHKERRQ(ierr);
            ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"implicit.h5", FILE_MODE_WRITE, &h5);CHKERRQ(ierr);
            ierr = PetscObjectSetName((PetscObject) uj, "uj");CHKERRQ(ierr);
            ierr = PetscObjectSetName((PetscObject) info, "info");CHKERRQ(ierr);
            ierr = VecView(info, h5);CHKERRQ(ierr);
            ierr = VecView(uj, h5);CHKERRQ(ierr);
            ierr = PetscViewerDestroy(&h5);CHKERRQ(ierr);
       }
    }
/* *********************************************** */
ierr = VecView(uj,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

ierr = VecDestroy(&info);CHKERRQ(ierr);
ierr = VecDestroy(&uj);CHKERRQ(ierr);
ierr = VecDestroy(&uj_old);CHKERRQ(ierr);
ierr = VecDestroy(&f);CHKERRQ(ierr);
ierr = MatDestroy(&A);CHKERRQ(ierr);

ierr = PetscFinalize();
return ierr;
}
