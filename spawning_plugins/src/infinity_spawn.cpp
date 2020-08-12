#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math.hh>


namespace gazebo {
    class InfinitySpawn : public ModelPlugin {
        public:
            void Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf) {
                this->_model = _parent;
                this->_update_connection = event::Events::ConnectWorldUpdateBegin(
                    std::bind(&InfinitySpawn::onUpdate, this)
                );

                _init_pose = this->_model->WorldPose();
                _init_time = common::Time::GetWallTime().Double();
            }

            void onUpdate() {
                double time = common::Time::GetWallTime().Double() - _init_time;
                ignition::math::Pose3d pose;
                pose = _init_pose;
                double scale = 2/(3 - std::cos(2*time)); // https://gamedev.stackexchange.com/questions/43691/how-can-i-move-an-object-in-an-infinity-or-figure-8-trajectory
                pose.Pos()[1] += scale*std::cos(time);
                pose.Pos()[2] += scale*std::sin(2*time)/2;
                this->_model->SetWorldPose(pose);
            }

        private:
            physics::ModelPtr _model;
            event::ConnectionPtr _update_connection;

            ignition::math::Pose3d _init_pose;
            double _init_time;
    };

    GZ_REGISTER_MODEL_PLUGIN(InfinitySpawn)
} // end of namespace gazebo
