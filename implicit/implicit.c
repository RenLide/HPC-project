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
    
    
/* ******************定义变量*********************** */
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
    
    
/* ******************初始化参数********************* */
    ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;    /*初始化Petsc*/
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,NULL,NULL);CHKERRQ(ierr);    /*开始读取选项参数*/
    ierr = PetscOptionsGetBool(NULL,NULL,"-r",&r,NULL);CHKERRQ(ierr);    /*从命令行读取是否重启（若有）*/
    ierr = PetscOptionsEnd();CHKERRQ(ierr);    /*读取选项参数结束*/
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);    /*设置并行MPI参数*/
/* *********************************************** */
    

/* *************计算并打印初始化变量信息*************** */
    dx = 1.0/GRID;    /*计算dx&dt*/
    dt = (t_end - t_start)/TSTEP;
    CFL= K*dt/(RHO*C*dx*dx);    /*计算CFL值*/
    ierr = PetscPrintf(PETSC_COMM_WORLD,"dx = %f\t",dx);CHKERRQ(ierr);    /*将dx的值打印出来，方便阅读输出文件时参考*/
    ierr = PetscPrintf(PETSC_COMM_WORLD,"dt = %f\t",dt);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"CFL = %f\n",CFL);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "grid = %d tstep = %d\n", GRID, TSTEP);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"restart is %d\n",r);CHKERRQ(ierr);
/* *********************************************** */
    
    
/* ******************初始化参数********************* */
    ierr = VecCreate(PETSC_COMM_WORLD,&uj);CHKERRQ(ierr);    /*创建一个并行空间*/
    ierr = VecCreate(PETSC_COMM_WORLD,&info);CHKERRQ(ierr);    /*创建临时向量*/
    ierr = VecSetSizes(uj,PETSC_DECIDE,GRID+1);CHKERRQ(ierr);    /*创建一个长度n+1的矩阵*/
    ierr = VecSetSizes(info, 3, PETSC_DECIDE);CHKERRQ(ierr);    /*创建长度为3的临时向量*/
    ierr = VecSetFromOptions(uj);CHKERRQ(ierr);    /*从选项数据库中配置向量*/
    ierr = VecSetFromOptions(info);CHKERRQ(ierr);    /*获得参数*/
    ierr = VecDuplicate(uj,&uj_old);CHKERRQ(ierr);    /*将x的格式赋给z*/
    ierr = VecDuplicate(uj,&f);CHKERRQ(ierr);    /*将x的格式赋给b*/

    ierr = VecGetOwnershipRange(uj,&START,&END);CHKERRQ(ierr);    /*设置并行x的起始终止点*/
    ierr = VecGetLocalSize(uj,&pos);CHKERRQ(ierr);    /*设置并行x的位置点*/

    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);    /*在并行空间创建一个矩阵*/
    ierr = MatSetSizes(A,pos,pos,GRID+1,GRID+1);CHKERRQ(ierr);    /*设置矩阵的行数和列数*/
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);    /*从选项数据库中配置矩阵*/
    ierr = MatSetUp(A);CHKERRQ(ierr);    /*开始建立矩阵*/
/* *********************************************** */

    
/* ****************初始化三对角矩阵****************** */

    if (!START){    /*若rstart为0时，即为首行*/
      START = 1;    /*将rstart设为1*/
      i = 0; col[0] = 0; col[1] = 1; coef[0] = 1+2.0*CFL; coef[1] = -CFL;    /*设置要用到的参数*/
      ierr   = MatSetValues(A,1,&i,2,col,coef,INSERT_VALUES);CHKERRQ(ierr);    /*设置三对角矩阵的第一行*/
    }
    
    if (END == GRID+1){    /*最后一行*/
      END = GRID;    /*将rend设为n*/
      i = GRID; col[0] = GRID-1; col[1] = GRID; coef[0] = -CFL; coef[1] = 1+2.0*CFL;    /*设置要用到的参数*/
      ierr = MatSetValues(A,1,&i,2,col,coef,INSERT_VALUES);CHKERRQ(ierr);    /*设置三对角矩阵的最后一行*/
    }

    coef[0] = -CFL; coef[1] = 1+2.0*CFL; coef[2] = -CFL;    /*设置三对角矩阵除首尾两行外的其余行的三个值*/
    for (i=START; i<END; i++){    /*除首尾两行外的行*/
      col[0] = i-1; col[1] = i; col[2] = i+1;    /*设置要用到的参数*/
      ierr = MatSetValues(A,1,&i,3,col,coef,INSERT_VALUES);CHKERRQ(ierr);    /*设置三对角矩阵的三对角值*/
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);    /*通知其余并行块将矩阵统一*/
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);    /*结束通知*/
    ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);    /*打印矩阵，检查是否出错*/
/* *********************************************** */
    
    
/* ***********根据-r参数对uj_old数据初始化************** */
    if(r){
        ierr = VecSet(uj_old, zero);CHKERRQ(ierr);    /*设置初始向量z*/
        if(rank == 0){    /*开始设置初始条件*/
          for(int i = 1; i < END; i++){    /*除首尾两个点外的其余点*/
            u0 = exp(i*dx);    /*根据当前位置来获取初始值*/
              ierr = VecSetValues(uj_old, 1, &i, &u0, INSERT_VALUES);CHKERRQ(ierr);    /*将向量的对应位置的值进行修改*/
          }
        }
        ierr = VecAssemblyBegin(uj_old);CHKERRQ(ierr);    /*通知其余并行块将向量统一*/
        ierr = VecAssemblyEnd(uj_old);CHKERRQ(ierr);    /*结束通知*/
    }
    else{
        ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"implicit.h5", FILE_MODE_READ, &h5);CHKERRQ(ierr);    /*创建输入文件*/
        ierr = PetscObjectSetName((PetscObject) uj_old, "uj");CHKERRQ(ierr);    /*将z输入的名字命名为explicit-vector*/
        ierr = PetscObjectSetName((PetscObject) info, "info");CHKERRQ(ierr);    /*将临时向量tem输入的名字命名为explicit-necess-data*/
        ierr = VecLoad(info, h5);CHKERRQ(ierr);    /*将读入的数据加载到向量tem中*/
        ierr = VecLoad(uj_old, h5);CHKERRQ(ierr);    /*将读入的数据加载到向量z中*/
        ierr = PetscViewerDestroy(&h5);CHKERRQ(ierr);    /*关闭输入*/
        index=0;    /*将索引初始化*/
        ierr = VecGetValues(info,1,&index,&dx);CHKERRQ(ierr);    /*将第一个值赋给dx*/
        index=index+1;    /*索引移向下一位*/
        ierr = VecGetValues(info,1,&index,&dt);CHKERRQ(ierr);    /*将第二个值赋给dt*/
        index=index+1;    /*索引移向下一位*/
        ierr = VecGetValues(info,1,&index,&t);CHKERRQ(ierr);    /*将第三个值赋给t*/
        index= 0;    /*索引复位*/
    }
    ierr = VecSet(f, zero);CHKERRQ(ierr);    /*设置初始向量b*/
    if(rank == 0){    /*开始设置初始条件*/
      for(i = 1; i < END; i++){    /*除首尾两个点外的其余点*/
        f0 = dt*sin(i*dx*PI);    /*根据当前位置来获取传热值*/
        ierr = VecSetValues(f, 1, &i, &f0, INSERT_VALUES);CHKERRQ(ierr);    /*将向量的对应位置的值进行修改*/
      }
    }
    ierr = VecAssemblyBegin(f);CHKERRQ(ierr);    /*通知其余并行块将向量统一*/
    ierr = VecAssemblyEnd(f);CHKERRQ(ierr);    /*结束通知*/
/* *********************************************** */
    
    
/* **************创建隐式求解ksp空间***************** */
    ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);    /*创建ksp解空间*/
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);    /*设置方程左侧的系数*/
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);    /*设置矩阵求解的相关系数*/
    ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);    /*设置pc的默认参数*/
    ierr = KSPSetTolerances(ksp,TOL,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);    /*设置各种误差值*/
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);    /*从选项数据库中配置ksp解空间*/
    
/* *********************************************** */
        
        
/* ****************迭代计算uj值********************* */
    while(PetscAbsReal(t)<=TEND){    /*计算0-2时间内的传播*/
       t += dt;    /*时间向前走*/
       ierr = VecAXPY(uj_old,1.0,f);CHKERRQ(ierr);    /*将求得的值加上在时间步长内的*/
       ierr = KSPSolve(ksp,uj_old,uj);CHKERRQ(ierr);    /*求解方程*/
       ierr = VecSetValues(uj, 1, &START, &zero, INSERT_VALUES);CHKERRQ(ierr);    /*设置边界条件*/
       ierr = VecSetValues(uj, 1, &END, &zero, INSERT_VALUES);CHKERRQ(ierr);    /*设置边界条件*/
       ierr = VecAssemblyBegin(uj);CHKERRQ(ierr);    /*统一向量更新*/
       ierr = VecAssemblyEnd(uj);CHKERRQ(ierr);    /*结束更新*/
       ierr = VecCopy(uj,uj_old);CHKERRQ(ierr);    /*将x的值赋给z*/
       iter += 1;    /*记录迭代次数*/
       if((iter % 10) == 0){    /*如果迭代次数为10的倍数，即每迭代十次*/
            index = 0;
            ierr = VecSetValues(info,1,&index,&dx,INSERT_VALUES);CHKERRQ(ierr);
            index += 1;
            ierr = VecSetValues(info,1,&index,&dt,INSERT_VALUES);CHKERRQ(ierr);
            index += 1;
            ierr = VecSetValues(info,1,&index,&t,INSERT_VALUES);CHKERRQ(ierr);
            index=0;
           
            ierr = VecAssemblyBegin(info);CHKERRQ(ierr);    /*通知其余并行块将向量统一*/
            ierr = VecAssemblyEnd(info);CHKERRQ(ierr);    /*结束通知*/

            ierr = PetscViewerCreate(PETSC_COMM_WORLD,&h5);CHKERRQ(ierr);    /*创建输出指针*/
            ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"implicit.h5", FILE_MODE_WRITE, &h5);CHKERRQ(ierr);    /*创建输出文件*/
            ierr = PetscObjectSetName((PetscObject) uj, "uj");CHKERRQ(ierr);    /*将z输出的名字命名为explicit-vector*/
            ierr = PetscObjectSetName((PetscObject) info, "info");CHKERRQ(ierr);    /*将tem输出的名字命名为explicit-necess-data*/
            ierr = VecView(info, h5);CHKERRQ(ierr);    /*tem输出到文件*/
            ierr = VecView(uj, h5);CHKERRQ(ierr);    /*z输出到文件*/
            ierr = PetscViewerDestroy(&h5);CHKERRQ(ierr);    /*关闭输出*/
       }
    }
/* *********************************************** */
ierr = VecView(uj,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);    /*打印向量，获得结束时显式方法的值*/

ierr = VecDestroy(&info);CHKERRQ(ierr);    /*关闭临时向量*/
ierr = VecDestroy(&uj);CHKERRQ(ierr);    /*关闭向量x*/
ierr = VecDestroy(&uj_old);CHKERRQ(ierr);    /*关闭向量z*/
ierr = VecDestroy(&f);CHKERRQ(ierr);    /*关闭向量b*/
ierr = MatDestroy(&A);CHKERRQ(ierr);    /*关闭矩阵A*/

ierr = PetscFinalize();    /*结束并行*/
return ierr;    /*程序结束*/
}
