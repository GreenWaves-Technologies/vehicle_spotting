
void print_tensor_checksum1(signed char * L2_tensor, signed char * L3_tensor, 
                            int TensDim, int TensCh, int printPartial){
  pi_cl_hyper_req_t req;
  int sum_par = 0;
  int sum_tot = 0;
  int sum_layer = 0;
  for(int ch=0;ch<TensCh;ch++){ 
    sum_tot = 0;     
    sum_par = 0;
    for(int i=0; i<TensDim;i++){
      pi_cl_ram_read(&HyperRam, L3_tensor+(i*TensDim+ch*TensDim*TensDim), 
        L2_tensor, TensDim*sizeof(signed char), &req);
      pi_cl_ram_read_wait(&req);
      sum_par = 0;
      for (int j=0; j<TensDim; j++){
//      	printf("%d ", (signed char) L2_tensor[j]);
        sum_par += (signed char) L2_tensor[j];
      }
//      printf("\n");
      sum_tot += sum_par;
    } 
    //printf("[Ch %d] Partial sum= %d\n",ch, sum_tot );
    sum_layer +=  sum_tot;
  }
  printf("Total sum= %d\n", sum_layer );
}

void print_checksum(signed char * tensor, int TensDim, int TensCh){
    int sum_tot = 0;
    for(int i= 0; i<TensDim*TensDim*TensCh; i++){
      sum_tot += tensor[i];
    }
    printf("checksum: %d\n",sum_tot );
}
