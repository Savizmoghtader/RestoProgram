using TrafficAssignment

#ta_data = load_ta_network("/home/juergen/Desktop/software/traffic_fw/XXX")
ta_data = load_ta_network("P:/NewCodes/traffic_fw/temp/XXX")
for i = 1:length(ta_data.capacity)
    if ta_data.capacity[i] == 0
        ta_data.capacity[i] == 100
    end
end



link_flow, link_travel_time, objective = ta_frank_wolfe(ta_data, method=:cfw, step=:newton, log=:on, tol=1e-4, max_iter_no=1000)

#flow = sum(link_flow)
#traveltime = sum(link_travel_time)
#car_hours = dot(link_travel_time, link_flow)
#car_distance = dot(ta_data.link_length,link_flow)

flow = 0.0

for i = 1:length(ta_data.capacity)
    if ta_data.link_type[i] != 9
        if ta_data.capacity[i] != 1
            global flow = flow + link_flow[i]
        end
    end
end

traveltime = 0.0
for i = 1:length(ta_data.capacity)
    if ta_data.link_type[i] != 9
        if ta_data.capacity[i] != 1
            global traveltime += link_travel_time[i]
        end
    end
end

car_hours = 0.0
for i = 1:length(ta_data.capacity)
    if ta_data.link_type[i] != 9
        if ta_data.capacity[i] != 0
            global car_hours += link_travel_time[i] * link_flow[i]
        end
    end
end

car_distance = 0.0
for i = 1:length(ta_data.capacity)
    if ta_data.link_type[i] != 9
        if ta_data.capacity[i] != 1
            global car_distance += ta_data.link_length[i] * link_flow[i]
        end
    end
end

println(flow," ",traveltime," ",car_hours," ",car_distance)


for i = 1:length(ta_data.capacity)
    if ta_data.capacity[i] == 1
        println(link_flow[i])
    end
end
